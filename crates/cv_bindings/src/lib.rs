//! `cv_bindings` — WebIDL ↔ Rust binding runtime.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum IdlType {
    Boolean,
    Byte,
    Octet,
    Short,
    UnsignedShort,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    Float,
    Double,
    DOMString,
    ByteString,
    USVString,
    Object,
    Any,
    Undefined,
    Null,
    Sequence(Box<IdlType>),
    Record(Box<IdlType>, Box<IdlType>),
    Promise(Box<IdlType>),
    Nullable(Box<IdlType>),
    Interface(String),
    Dictionary(String),
    Enum(String),
    Callback(String),
}

#[derive(Debug, Clone)]
pub struct IdlArgument {
    pub name: String,
    pub ty: IdlType,
    pub optional: bool,
    pub default: Option<IdlDefault>,
    pub variadic: bool,
}

#[derive(Debug, Clone)]
pub enum IdlDefault {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    EmptySequence,
    EmptyDictionary,
}

#[derive(Debug, Clone)]
pub struct IdlMethod {
    pub name: String,
    pub args: Vec<IdlArgument>,
    pub return_ty: IdlType,
    pub is_static: bool,
}

#[derive(Debug, Clone)]
pub struct IdlAttribute {
    pub name: String,
    pub ty: IdlType,
    pub readonly: bool,
    pub is_static: bool,
}

#[derive(Debug, Clone, Default)]
pub struct IdlInterface {
    pub name: String,
    pub inherits: Option<String>,
    pub methods: Vec<IdlMethod>,
    pub attributes: Vec<IdlAttribute>,
}

#[derive(Debug, Default)]
pub struct BindingRegistry {
    interfaces: HashMap<String, IdlInterface>,
}

impl BindingRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, iface: IdlInterface) {
        self.interfaces.insert(iface.name.clone(), iface);
    }
    pub fn get(&self, name: &str) -> Option<&IdlInterface> {
        self.interfaces.get(name)
    }
    pub fn method(&self, iface_name: &str, method: &str) -> Option<&IdlMethod> {
        let mut cur = self.get(iface_name)?;
        loop {
            if let Some(m) = cur.methods.iter().find(|m| m.name == method) {
                return Some(m);
            }
            let parent = cur.inherits.as_ref()?;
            cur = self.get(parent)?;
        }
    }
    pub fn attribute(&self, iface_name: &str, attr: &str) -> Option<&IdlAttribute> {
        let mut cur = self.get(iface_name)?;
        loop {
            if let Some(a) = cur.attributes.iter().find(|a| a.name == attr) {
                return Some(a);
            }
            let parent = cur.inherits.as_ref()?;
            cur = self.get(parent)?;
        }
    }
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }
}

/// Validate JS-side argument list against an IDL signature. Reports
/// arity mismatch + per-arg type-tag mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsType {
    Undefined,
    Null,
    Bool,
    Number,
    String,
    Object,
    Array,
    Function,
}

pub fn validate_call(method: &IdlMethod, given: &[JsType]) -> Result<(), BindingError> {
    let required = method
        .args
        .iter()
        .filter(|a| !a.optional && !a.variadic)
        .count();
    if given.len() < required {
        return Err(BindingError::Arity {
            expected: required,
            got: given.len(),
        });
    }
    for (i, arg) in method.args.iter().enumerate() {
        let g = match given.get(i) {
            Some(g) => *g,
            None => break,
        };
        if !js_compatible(&arg.ty, g) {
            return Err(BindingError::ArgType {
                index: i,
                expected: arg.ty.clone(),
                got: g,
            });
        }
    }
    Ok(())
}

fn js_compatible(idl: &IdlType, js: JsType) -> bool {
    match idl {
        IdlType::Boolean => matches!(js, JsType::Bool | JsType::Number),
        IdlType::Byte
        | IdlType::Octet
        | IdlType::Short
        | IdlType::UnsignedShort
        | IdlType::Long
        | IdlType::UnsignedLong
        | IdlType::LongLong
        | IdlType::UnsignedLongLong
        | IdlType::Float
        | IdlType::Double => matches!(js, JsType::Number),
        IdlType::DOMString | IdlType::ByteString | IdlType::USVString => true, // coerce-from-any
        IdlType::Object | IdlType::Dictionary(_) => matches!(js, JsType::Object | JsType::Null),
        IdlType::Any => true,
        IdlType::Sequence(_) => matches!(js, JsType::Array | JsType::Object),
        IdlType::Nullable(inner) => js == JsType::Null || js_compatible(inner, js),
        IdlType::Callback(_) => matches!(js, JsType::Function),
        IdlType::Interface(_) => matches!(js, JsType::Object | JsType::Null),
        IdlType::Promise(_) | IdlType::Record(_, _) => matches!(js, JsType::Object),
        IdlType::Undefined => matches!(js, JsType::Undefined),
        IdlType::Null => matches!(js, JsType::Null),
        IdlType::Enum(_) => matches!(js, JsType::String),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BindingError {
    UnknownInterface(String),
    UnknownMember(String),
    Arity {
        expected: usize,
        got: usize,
    },
    ArgType {
        index: usize,
        expected: IdlType,
        got: JsType,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_iface() -> IdlInterface {
        IdlInterface {
            name: "Document".into(),
            inherits: Some("Node".into()),
            methods: vec![IdlMethod {
                name: "createElement".into(),
                args: vec![IdlArgument {
                    name: "tagName".into(),
                    ty: IdlType::DOMString,
                    optional: false,
                    default: None,
                    variadic: false,
                }],
                return_ty: IdlType::Interface("Element".into()),
                is_static: false,
            }],
            attributes: vec![IdlAttribute {
                name: "URL".into(),
                ty: IdlType::USVString,
                readonly: true,
                is_static: false,
            }],
        }
    }

    #[test]
    fn registry_lookup_method() {
        let mut r = BindingRegistry::new();
        r.register(make_iface());
        let m = r.method("Document", "createElement").unwrap();
        assert_eq!(m.name, "createElement");
        assert_eq!(m.args.len(), 1);
    }

    #[test]
    fn method_lookup_follows_inheritance() {
        let mut r = BindingRegistry::new();
        let mut node = IdlInterface::default();
        node.name = "Node".into();
        node.methods.push(IdlMethod {
            name: "appendChild".into(),
            args: vec![IdlArgument {
                name: "child".into(),
                ty: IdlType::Interface("Node".into()),
                optional: false,
                default: None,
                variadic: false,
            }],
            return_ty: IdlType::Interface("Node".into()),
            is_static: false,
        });
        r.register(node);
        r.register(make_iface());
        let m = r.method("Document", "appendChild");
        assert!(m.is_some(), "inherited method should be found");
    }

    #[test]
    fn validate_call_arity() {
        let iface = make_iface();
        let m = &iface.methods[0];
        assert!(validate_call(m, &[JsType::String]).is_ok());
        let err = validate_call(m, &[]).unwrap_err();
        assert!(matches!(
            err,
            BindingError::Arity {
                expected: 1,
                got: 0
            }
        ));
    }

    #[test]
    fn validate_call_type_mismatch() {
        // DOMString coerces from anything (WebIDL ToString), so use
        // an Interface arg for a non-coercing constraint.
        let m2 = IdlMethod {
            name: "x".into(),
            args: vec![IdlArgument {
                name: "n".into(),
                ty: IdlType::Long,
                optional: false,
                default: None,
                variadic: false,
            }],
            return_ty: IdlType::Undefined,
            is_static: false,
        };
        assert!(matches!(
            validate_call(&m2, &[JsType::String]),
            Err(BindingError::ArgType { .. })
        ));
    }

    #[test]
    fn nullable_accepts_null_or_inner_type() {
        let m = IdlMethod {
            name: "x".into(),
            args: vec![IdlArgument {
                name: "n".into(),
                ty: IdlType::Nullable(Box::new(IdlType::Long)),
                optional: false,
                default: None,
                variadic: false,
            }],
            return_ty: IdlType::Undefined,
            is_static: false,
        };
        assert!(validate_call(&m, &[JsType::Null]).is_ok());
        assert!(validate_call(&m, &[JsType::Number]).is_ok());
        assert!(validate_call(&m, &[JsType::Bool]).is_err());
    }
}
