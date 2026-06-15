//! UI Automation provider — IRawElementProviderSimple +
//! IRawElementProviderFragment + IRawElementProviderFragmentRoot.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use crate::{AxNode, AxRole, AxTree};

// ----- Win32 GUIDs ----------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GUID {
    pub Data1: u32,
    pub Data2: u16,
    pub Data3: u16,
    pub Data4: [u8; 8],
}

/// IID_IUnknown: 00000000-0000-0000-C000-000000000046
pub const IID_IUNKNOWN: GUID = GUID {
    Data1: 0x00000000,
    Data2: 0x0000,
    Data3: 0x0000,
    Data4: [0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
};
/// IID_IRawElementProviderSimple: D6DD68D1-86FD-4332-8666-9ABEDEA2D24C
pub const IID_IRAW_ELEMENT_PROVIDER_SIMPLE: GUID = GUID {
    Data1: 0xD6DD68D1,
    Data2: 0x86FD,
    Data3: 0x4332,
    Data4: [0x86, 0x66, 0x9A, 0xBE, 0xDE, 0xA2, 0xD2, 0x4C],
};
/// IID_IRawElementProviderFragment: F7063DA8-8359-439C-9297-BBC5299A7D87
pub const IID_IRAW_ELEMENT_PROVIDER_FRAGMENT: GUID = GUID {
    Data1: 0xF7063DA8,
    Data2: 0x8359,
    Data3: 0x439C,
    Data4: [0x92, 0x97, 0xBB, 0xC5, 0x29, 0x9A, 0x7D, 0x87],
};
/// IID_IRawElementProviderFragmentRoot: 620CE2A5-AB8F-40A9-86CB-DE3C75599B58
pub const IID_IRAW_ELEMENT_PROVIDER_FRAGMENT_ROOT: GUID = GUID {
    Data1: 0x620CE2A5,
    Data2: 0xAB8F,
    Data3: 0x40A9,
    Data4: [0x86, 0xCB, 0xDE, 0x3C, 0x75, 0x59, 0x9B, 0x58],
};

// ----- UI Automation enums --------------------------------------------

/// `UIA_ControlTypeIds` — published values from UIAutomationCoreApi.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UiaControlType {
    Button = 50000,
    Calendar = 50001,
    CheckBox = 50002,
    ComboBox = 50003,
    Edit = 50004,
    Hyperlink = 50005,
    Image = 50006,
    ListItem = 50007,
    List = 50008,
    Menu = 50009,
    MenuBar = 50010,
    MenuItem = 50011,
    ProgressBar = 50012,
    RadioButton = 50013,
    ScrollBar = 50014,
    Slider = 50015,
    Spinner = 50016,
    StatusBar = 50017,
    Tab = 50018,
    TabItem = 50019,
    Text = 50020,
    ToolBar = 50021,
    ToolTip = 50022,
    Tree = 50023,
    TreeItem = 50024,
    Custom = 50025,
    Group = 50026,
    Thumb = 50027,
    DataGrid = 50028,
    DataItem = 50029,
    Document = 50030,
    SplitButton = 50031,
    Window = 50032,
    Pane = 50033,
    Header = 50034,
    HeaderItem = 50035,
    Table = 50036,
    TitleBar = 50037,
    Separator = 50038,
}

/// `UIA_PropertyIds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UiaPropertyId {
    RuntimeId = 30000,
    BoundingRectangle = 30001,
    ProcessId = 30002,
    ControlType = 30003,
    LocalizedControlType = 30004,
    Name = 30005,
    AcceleratorKey = 30006,
    AccessKey = 30007,
    HasKeyboardFocus = 30008,
    IsKeyboardFocusable = 30009,
    IsEnabled = 30010,
    AutomationId = 30011,
    ClassName = 30012,
    HelpText = 30013,
    ClickablePoint = 30014,
    Culture = 30015,
    IsControlElement = 30016,
    IsContentElement = 30017,
    LabeledBy = 30018,
    IsPassword = 30019,
    NativeWindowHandle = 30020,
    ItemType = 30021,
    IsOffscreen = 30022,
    Orientation = 30023,
    FrameworkId = 30024,
    IsRequiredForForm = 30025,
    // Pattern-property ids (UIAutomationCoreApi.h):
    ToggleToggleState = 30086,
    ExpandCollapseState = 30070,
    ValueValue = 30045,
    ValueIsReadOnly = 30046,
    /// `UIA_LevelPropertyId` — heading / hierarchy level.
    Level = 30154,
    /// `UIA_AriaRolePropertyId` — the raw ARIA role string (e.g. "button").
    AriaRole = 30101,
}

/// `NavigateDirection` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NavigateDirection {
    Parent = 0,
    NextSibling = 1,
    PreviousSibling = 2,
    FirstChild = 3,
    LastChild = 4,
}

/// `ProviderOptions` bitmask.
pub const PROVIDER_OPTIONS_SERVER_SIDE_PROVIDER: i32 = 1;
pub const PROVIDER_OPTIONS_CLIENT_SIDE_PROVIDER: i32 = 2;
pub const PROVIDER_OPTIONS_NON_CLIENT_AREA_PROVIDER: i32 = 4;
pub const PROVIDER_OPTIONS_OVERRIDE_PROVIDER: i32 = 8;
pub const PROVIDER_OPTIONS_PROVIDER_OWNS_SET_FOCUS: i32 = 16;
pub const PROVIDER_OPTIONS_PROVIDER_OWNS_BOUNDING_RECTANGLE: i32 = 32;

/// `UIAFRAGMENT_ROOT` const — indicates the runtime-id sentinel
/// IRawElementProviderFragmentRoot::get_RuntimeId returns when the
/// element is the fragment root.
pub const UIAFRAGMENT_ROOT: i32 = 3;

// ----- Pure-Rust provider model ---------------------------------------

/// Mirror of the UIA `ToggleState` enum (`UIA_ToggleStateIds`) returned by the
/// Toggle control pattern's `get_ToggleState`. Off/On/Indeterminate match the
/// checkbox/switch tri-state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ToggleState {
    Off = 0,
    On = 1,
    Indeterminate = 2,
    /// Sentinel: the element does not support the Toggle pattern at all.
    Unsupported = -1,
}

#[derive(Debug, Clone)]
pub struct UiaProvider {
    pub runtime_id: Vec<i32>,
    pub control_type: UiaControlType,
    pub name: String,
    pub help_text: String,
    /// Value pattern's current value (UIA `ValuePattern.Value`).
    pub value: String,
    pub bounding: (f64, f64, f64, f64), // l, t, w, h
    pub is_keyboard_focusable: bool,
    pub has_keyboard_focus: bool,
    pub is_enabled: bool,
    pub is_password: bool,
    /// `aria-required` / native `required` (UIA `IsRequiredForForm`).
    pub is_required: bool,
    /// Toggle pattern state for checkbox/radio/switch (Unsupported otherwise).
    pub toggle_state: ToggleState,
    /// ExpandCollapse pattern state: `None` when not a disclosure widget.
    pub expand_collapse: Option<bool>,
    /// Heading / tree depth level for the `Level` property (0 when N/A).
    pub level: u32,
    pub automation_id: String,
    pub class_name: String,
    pub framework_id: String,
}

impl UiaProvider {
    pub fn from_ax(node: &AxNode) -> Self {
        use crate::{CheckedState, ExpandedState};
        let toggle_state = match node.checked {
            CheckedState::None => ToggleState::Unsupported,
            CheckedState::Unchecked => ToggleState::Off,
            CheckedState::Checked => ToggleState::On,
            CheckedState::Mixed => ToggleState::Indeterminate,
        };
        let expand_collapse = match node.expanded {
            ExpandedState::Undefined => None,
            ExpandedState::Collapsed => Some(false),
            ExpandedState::Expanded => Some(true),
        };
        Self {
            runtime_id: vec![UIAFRAGMENT_ROOT, node.id as i32],
            control_type: control_type_for_role(node.role),
            name: node.name.clone(),
            help_text: node.description.clone(),
            value: node.value.clone(),
            bounding: (
                node.bbox.0 as f64,
                node.bbox.1 as f64,
                node.bbox.2 as f64,
                node.bbox.3 as f64,
            ),
            is_keyboard_focusable: matches!(
                node.role,
                AxRole::Button
                    | AxRole::Textbox
                    | AxRole::Searchbox
                    | AxRole::Combobox
                    | AxRole::Link
                    | AxRole::Checkbox
                    | AxRole::Radio
                    | AxRole::Slider
                    | AxRole::Spinbutton
                    | AxRole::Tab
            ),
            has_keyboard_focus: node.focused,
            is_enabled: !node.disabled,
            is_password: node.password,
            is_required: node.required,
            toggle_state,
            expand_collapse,
            level: node.level,
            automation_id: format!("ax-{}", node.id),
            class_name: format!("{:?}", node.role),
            framework_id: "Conclave".into(),
        }
    }

    /// Implements `IRawElementProviderSimple::GetPropertyValue(propertyId)`
    /// — returns a tagged variant (VARIANT in COM; here we use an
    /// enum that the COM glue translates).
    pub fn property(&self, id: UiaPropertyId) -> VariantValue {
        use UiaPropertyId::*;
        match id {
            ControlType => VariantValue::I4(self.control_type as i32),
            Name => VariantValue::BStr(self.name.clone()),
            HelpText => VariantValue::BStr(self.help_text.clone()),
            BoundingRectangle => VariantValue::R8Array(vec![
                self.bounding.0,
                self.bounding.1,
                self.bounding.2,
                self.bounding.3,
            ]),
            HasKeyboardFocus => VariantValue::Bool(self.has_keyboard_focus),
            IsKeyboardFocusable => VariantValue::Bool(self.is_keyboard_focusable),
            IsEnabled => VariantValue::Bool(self.is_enabled),
            IsPassword => VariantValue::Bool(self.is_password),
            IsRequiredForForm => VariantValue::Bool(self.is_required),
            ToggleToggleState => VariantValue::I4(self.toggle_state as i32),
            ExpandCollapseState => VariantValue::I4(match self.expand_collapse {
                // ExpandCollapseState: Collapsed=0, Expanded=1, LeafNode=3.
                Some(true) => 1,
                Some(false) => 0,
                None => 3,
            }),
            ValueValue => VariantValue::BStr(self.value.clone()),
            ValueIsReadOnly => VariantValue::Bool(false),
            Level => VariantValue::I4(self.level as i32),
            AriaRole => VariantValue::BStr(aria_role_string(self.control_type).to_string()),
            AutomationId => VariantValue::BStr(self.automation_id.clone()),
            ClassName => VariantValue::BStr(self.class_name.clone()),
            FrameworkId => VariantValue::BStr(self.framework_id.clone()),
            RuntimeId => VariantValue::I4Array(self.runtime_id.clone()),
            IsContentElement | IsControlElement => VariantValue::Bool(true),
            IsOffscreen => VariantValue::Bool(false),
            _ => VariantValue::Empty,
        }
    }

    /// Report which UIA control patterns this provider supports — used by the
    /// COM bridge's `GetPatternProvider`. A screen reader queries these to know
    /// it may call `Toggle()`/`Expand()`/`get_Value()` etc.
    pub fn supports_pattern(&self, pattern: UiaPatternId) -> bool {
        use UiaPatternId::*;
        match pattern {
            Invoke => matches!(
                self.control_type,
                UiaControlType::Button | UiaControlType::Hyperlink | UiaControlType::MenuItem
            ),
            Toggle => self.toggle_state != ToggleState::Unsupported,
            ExpandCollapse => self.expand_collapse.is_some(),
            Value => matches!(
                self.control_type,
                UiaControlType::Edit | UiaControlType::ComboBox
            ),
            RangeValue => matches!(
                self.control_type,
                UiaControlType::Slider | UiaControlType::Spinner | UiaControlType::ProgressBar
            ),
            SelectionItem => matches!(
                self.control_type,
                UiaControlType::RadioButton | UiaControlType::TabItem | UiaControlType::ListItem
            ),
            Selection => matches!(
                self.control_type,
                UiaControlType::List | UiaControlType::ComboBox | UiaControlType::Tab
            ),
        }
    }

    pub fn provider_options(&self) -> i32 {
        PROVIDER_OPTIONS_SERVER_SIDE_PROVIDER
    }
}

/// Subset of COM VARIANT used by UIA's GetPropertyValue.
#[derive(Debug, Clone, PartialEq)]
pub enum VariantValue {
    Empty,
    Bool(bool),
    I4(i32),
    R8(f64),
    BStr(String),
    I4Array(Vec<i32>),
    R8Array(Vec<f64>),
}

/// `UIA_PatternIds` subset — the control patterns we can fulfil from the AX
/// model. Values from UIAutomationCoreApi.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UiaPatternId {
    Invoke = 10000,
    Selection = 10001,
    Value = 10002,
    RangeValue = 10003,
    SelectionItem = 10010,
    ExpandCollapse = 10005,
    Toggle = 10015,
}

/// The ARIA role token a UIA control type round-trips to for the
/// `AriaRolePropertyId`. Best-effort: UIA control types are coarser than ARIA
/// roles, so this is the canonical role for the type.
fn aria_role_string(ct: UiaControlType) -> &'static str {
    match ct {
        UiaControlType::Button => "button",
        UiaControlType::CheckBox => "checkbox",
        UiaControlType::ComboBox => "combobox",
        UiaControlType::Edit => "textbox",
        UiaControlType::Hyperlink => "link",
        UiaControlType::Image => "img",
        UiaControlType::ListItem => "listitem",
        UiaControlType::List => "list",
        UiaControlType::Menu => "menu",
        UiaControlType::MenuItem => "menuitem",
        UiaControlType::RadioButton => "radio",
        UiaControlType::Slider => "slider",
        UiaControlType::Spinner => "spinbutton",
        UiaControlType::Tab => "tablist",
        UiaControlType::TabItem => "tab",
        UiaControlType::Text => "heading",
        UiaControlType::Tree => "tree",
        UiaControlType::TreeItem => "treeitem",
        UiaControlType::Table => "table",
        UiaControlType::Group => "group",
        UiaControlType::Document => "document",
        _ => "",
    }
}

fn control_type_for_role(role: AxRole) -> UiaControlType {
    match role {
        AxRole::Button => UiaControlType::Button,
        AxRole::Checkbox => UiaControlType::CheckBox,
        AxRole::Combobox => UiaControlType::ComboBox,
        AxRole::Textbox | AxRole::Searchbox => UiaControlType::Edit,
        AxRole::Link => UiaControlType::Hyperlink,
        AxRole::Image => UiaControlType::Image,
        AxRole::Document => UiaControlType::Document,
        AxRole::Heading | AxRole::Paragraph => UiaControlType::Text,
        AxRole::List => UiaControlType::List,
        AxRole::ListItem => UiaControlType::ListItem,
        AxRole::Menu => UiaControlType::Menu,
        AxRole::MenuItem => UiaControlType::MenuItem,
        AxRole::Radio => UiaControlType::RadioButton,
        AxRole::Slider => UiaControlType::Slider,
        AxRole::Spinbutton => UiaControlType::Spinner,
        AxRole::Status => UiaControlType::StatusBar,
        AxRole::Tab => UiaControlType::TabItem,
        AxRole::Tablist => UiaControlType::Tab,
        AxRole::Tabpanel => UiaControlType::Pane,
        AxRole::Tree => UiaControlType::Tree,
        AxRole::TreeItem => UiaControlType::TreeItem,
        AxRole::Group | AxRole::Section | AxRole::Article | AxRole::Region => UiaControlType::Group,
        AxRole::Table => UiaControlType::Table,
        AxRole::Form => UiaControlType::Group,
        AxRole::Main
        | AxRole::Navigation
        | AxRole::Banner
        | AxRole::Contentinfo
        | AxRole::Complementary => UiaControlType::Pane,
        AxRole::Search => UiaControlType::Group,
        AxRole::Application => UiaControlType::Window,
        // Presentational nodes are pruned before reaching UIA; if one somehow
        // surfaces, expose it as a structureless Group rather than a fake leaf.
        AxRole::Presentation | AxRole::Generic => UiaControlType::Custom,
    }
}

/// Navigate from `from` in the given direction within the AxTree.
pub fn navigate(tree: &AxTree, from: u32, dir: NavigateDirection) -> Option<u32> {
    let n = tree.get(from)?;
    match dir {
        NavigateDirection::Parent => n.parent,
        NavigateDirection::FirstChild => n.children.first().copied(),
        NavigateDirection::LastChild => n.children.last().copied(),
        NavigateDirection::NextSibling => {
            let p = tree.get(n.parent?)?;
            let idx = p.children.iter().position(|&c| c == from)?;
            p.children.get(idx + 1).copied()
        }
        NavigateDirection::PreviousSibling => {
            let p = tree.get(n.parent?)?;
            let idx = p.children.iter().position(|&c| c == from)?;
            if idx == 0 {
                None
            } else {
                p.children.get(idx - 1).copied()
            }
        }
    }
}

/// Win32 UIAutomationCore.dll exports we need to register a provider
/// with the OS.
#[link(name = "UIAutomationCore")]
unsafe extern "system" {
    pub fn UiaReturnRawElementProvider(
        hwnd: *mut std::ffi::c_void,
        wParam: usize,
        lParam: isize,
        el: *mut std::ffi::c_void,
    ) -> isize;
    pub fn UiaHostProviderFromHwnd(
        hwnd: *mut std::ffi::c_void,
        ppProvider: *mut *mut std::ffi::c_void,
    ) -> i32;
    pub fn UiaRaiseAutomationEvent(pProvider: *mut std::ffi::c_void, id: i32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_returns_control_type() {
        let mut t = AxTree::new();
        let id = t.add(None, AxRole::Button, "OK");
        let provider = UiaProvider::from_ax(t.get(id).unwrap());
        assert_eq!(
            provider.property(UiaPropertyId::ControlType),
            VariantValue::I4(50000)
        );
    }

    #[test]
    fn property_returns_name_as_bstr() {
        let mut t = AxTree::new();
        let id = t.add(None, AxRole::Button, "Save");
        let provider = UiaProvider::from_ax(t.get(id).unwrap());
        assert_eq!(
            provider.property(UiaPropertyId::Name),
            VariantValue::BStr("Save".into())
        );
    }

    #[test]
    fn property_bounding_returns_4_doubles() {
        let mut t = AxTree::new();
        let id = t.add(None, AxRole::Button, "x");
        let provider = UiaProvider::from_ax(t.get(id).unwrap());
        match provider.property(UiaPropertyId::BoundingRectangle) {
            VariantValue::R8Array(v) => assert_eq!(v.len(), 4),
            _ => panic!("expected R8Array"),
        }
    }

    #[test]
    fn navigate_first_child_walks_down() {
        let mut t = AxTree::new();
        let parent = t.add(None, AxRole::Main, "main");
        let child = t.add(Some(parent), AxRole::Button, "btn");
        assert_eq!(
            navigate(&t, parent, NavigateDirection::FirstChild),
            Some(child)
        );
        assert_eq!(navigate(&t, child, NavigateDirection::Parent), Some(parent));
    }

    #[test]
    fn navigate_siblings() {
        let mut t = AxTree::new();
        let parent = t.add(None, AxRole::Main, "main");
        let a = t.add(Some(parent), AxRole::Button, "a");
        let b = t.add(Some(parent), AxRole::Button, "b");
        let c = t.add(Some(parent), AxRole::Button, "c");
        assert_eq!(navigate(&t, a, NavigateDirection::NextSibling), Some(b));
        assert_eq!(navigate(&t, c, NavigateDirection::PreviousSibling), Some(b));
        assert!(navigate(&t, a, NavigateDirection::PreviousSibling).is_none());
    }

    #[test]
    fn runtime_id_includes_fragment_root_sentinel() {
        let mut t = AxTree::new();
        let id = t.add(None, AxRole::Button, "x");
        let provider = UiaProvider::from_ax(t.get(id).unwrap());
        assert_eq!(provider.runtime_id[0], UIAFRAGMENT_ROOT);
    }

    #[test]
    fn keyboard_focusable_for_interactive_roles() {
        let mut t = AxTree::new();
        let btn = t.add(None, AxRole::Button, "x");
        let label = t.add(None, AxRole::Paragraph, "y");
        let pb = UiaProvider::from_ax(t.get(btn).unwrap());
        let pl = UiaProvider::from_ax(t.get(label).unwrap());
        assert!(pb.is_keyboard_focusable);
        assert!(!pl.is_keyboard_focusable);
    }

    #[test]
    fn control_type_for_link_role() {
        let mut t = AxTree::new();
        let id = t.add(None, AxRole::Link, "go");
        let p = UiaProvider::from_ax(t.get(id).unwrap());
        assert_eq!(p.control_type, UiaControlType::Hyperlink);
    }
}
