
"""
Structural Model Builder - Professional Streamlit App
Clean, modern interface with example management
"""

import numpy as np
import streamlit as st
import os
import shutil
import zipfile
import json
from pathlib import Path
from io import BytesIO
from datetime import datetime


def create_regular_polygon_nodes(center_x, center_y, radius, n_sides, start_id, z=0.0):
    """Create regular polygon nodes dictionary"""
    angles = np.linspace(0, 2*np.pi, n_sides + 1)[:-1]
    nodes = {}
    for i, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        nodes[start_id + i] = (x, y, z)
    return nodes


def patch_gmsh():
    """Fix GMSH signal handling"""
    import signal as sig
    orig = sig.signal
    def dummy(sn, h):
        try:
            return orig(sn, h)
        except ValueError:
            return None
    sig.signal = dummy
    return orig


def validate_config(config_text):
    """Validate configuration syntax"""
    try:
        compile(config_text, '<string>', 'exec')
        return True, "Valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def save_examples_to_file(examples):
    """Save examples to JSON file"""
    examples_file = Path("saved_examples.json")
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)


def load_examples_from_file():
    """Load examples from JSON file"""
    examples_file = Path("saved_examples.json")
    if examples_file.exists():
        try:
            with open(examples_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def get_default_examples():
    """Get default example configurations"""
    examples = {}
    
    examples['simple_building'] = {
        'name': 'Simple Building',
        'description': '2-column structure with fiber sections',
        'code': '''# Simple Building Example
import numpy as np
from test3 import build_model

materials = {
    'concrete_cover': {'elastic_modulus': 30e9, 'poissons_ratio': 0.2, 'density': 2400, 'color': '#dbb40c'},
    'concrete_core': {'elastic_modulus': 30e9, 'poissons_ratio': 0.2, 'density': 2400, 'color': '#88b378'},
    'steel_rebar': {'elastic_modulus': 200e9, 'poissons_ratio': 0.3, 'density': 7850, 'yield_strength': 500e6, 'color': 'black'}
}

outline = [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]
rebar = [{'type': 'points', 'points': [[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]], 
          'dia': 0.02, 'color': 'black', 'group_name': 'Main_Rebars'}]

node_coords = {1: (0, 0, 0), 2: (5, 0, 0), 3: (0, 0, 3), 4: (5, 0, 3)}
boundary_conditions = {1: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1]}

element_configs = {
    'transformations': [{'type': 'Linear', 'tag': 1, 'vecxz': [0, 1, 0]}],
    'integrations': [{'type': 'Lobatto', 'tag': 1, 'sec_tag': 1, 'np': 5}],
    'force_beam_columns': [
        {'tag': 1, 'node_i': 1, 'node_j': 3, 'transf_tag': 1, 'integ_tag': 1},
        {'tag': 2, 'node_i': 2, 'node_j': 4, 'transf_tag': 1, 'integ_tag': 1}
    ],
    'elastic_beam_columns': []
}

material_params = [
    ['Concrete01', 1, -30e6, -0.002, -6e6, -0.006],
    ['Concrete01', 2, -30e6, -0.002, -6e6, -0.006],
    ['Steel01', 3, 500e6, 200e9, 0.01]
]

results = build_model(
    model_params={'ndm': 3, 'ndf': 6},
    materials_list=[materials],
    outline_points_list=[outline],
    rebar_configs_list=[rebar],
    section_params_list=[{'cover': 0.05, 'mesh_size': 0.05, 'mat_tags': {'cover': 1, 'core': 2, 'rebar': 3},
                          'sec_tag': 1, 'G': 12.5e9, 'save_prefix': 'section_1', 'section_name': 'Column_Section'}],
    material_params=material_params,
    node_coords=node_coords,
    boundary_conditions=boundary_conditions,
    element_configs=element_configs,
    spring_configs=None,
    nodal_spring_configs=None,
    start_base_node_id=10000000,
    diaphragm_list=None,
    start_node_id=20000,
    start_element_id=20000,
    load_configs=None,
    mass_configs=None,
    visualize=True,
    output_dir="output",
    slab_configs=None,
    existing_frame_nodes=None
)
'''
    }
    
    examples['frame_springs'] = {
        'name': 'Frame with Springs',
        'description': 'Frame structure with zero-length spring supports',
        'code': '''# Frame with Springs Example
import numpy as np
from test3 import build_model

materials = {
    'concrete_cover': {'elastic_modulus': 30e9, 'poissons_ratio': 0.2, 'density': 2400, 'color': '#dbb40c'},
    'concrete_core': {'elastic_modulus': 30e9, 'poissons_ratio': 0.2, 'density': 2400, 'color': '#88b378'},
    'steel_rebar': {'elastic_modulus': 200e9, 'poissons_ratio': 0.3, 'density': 7850, 'yield_strength': 500e6, 'color': 'black'}
}

outline = [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]]
rebar = [{'type': 'points', 'points': [[-0.35, -0.35], [0.35, -0.35], [0.35, 0.35], [-0.35, 0.35]], 
          'dia': 0.025, 'color': 'black', 'group_name': 'Rebars'}]

node_coords = {1: (0, 0, 0), 2: (6, 0, 0), 3: (0, 0, 4), 4: (6, 0, 4)}
boundary_conditions = {}

element_configs = {
    'transformations': [{'type': 'Linear', 'tag': 1, 'vecxz': [0, 1, 0]}],
    'integrations': [{'type': 'Lobatto', 'tag': 1, 'sec_tag': 1, 'np': 5}],
    'force_beam_columns': [
        {'tag': 1, 'node_i': 1, 'node_j': 3, 'transf_tag': 1, 'integ_tag': 1},
        {'tag': 2, 'node_i': 2, 'node_j': 4, 'transf_tag': 1, 'integ_tag': 1}
    ],
    'elastic_beam_columns': [
        {'tag': 3, 'node_i': 3, 'node_j': 4, 'A': 0.64, 'E': 30e9, 'G': 12.5e9, 'J': 0.03413, 'Iy': 0.01365, 'Iz': 0.01365, 'transf_tag': 1}
    ]
}

material_params = [
    ['Concrete01', 1, -30e6, -0.002, -6e6, -0.006],
    ['Concrete01', 2, -30e6, -0.002, -6e6, -0.006],
    ['Steel01', 3, 500e6, 200e9, 0.01],
    ['Elastic', 4, 1e8]
]

nodal_spring_configs = {
    'material_props': {'id': 4, 'directions': [1, 2, 3], 'config': ['Elastic', 4, 1e8]},
    'node_list': [(1, 0, 0, 0), (2, 6, 0, 0)],
    'boundary_condition': [1, 1, 1, 1, 1, 1],
    'element_start_id': 100000,
    'spring_node_start_id': 10000000
}

results = build_model(
    model_params={'ndm': 3, 'ndf': 6},
    materials_list=[materials],
    outline_points_list=[outline],
    rebar_configs_list=[rebar],
    section_params_list=[{'cover': 0.05, 'mesh_size': 0.05, 'mat_tags': {'cover': 1, 'core': 2, 'rebar': 3},
                          'sec_tag': 1, 'G': 12.5e9, 'save_prefix': 'section_springs', 'section_name': 'Column_Springs'}],
    material_params=material_params,
    node_coords=node_coords,
    boundary_conditions=boundary_conditions,
    element_configs=element_configs,
    spring_configs=None,
    nodal_spring_configs=nodal_spring_configs,
    start_base_node_id=10000000,
    diaphragm_list=None,
    start_node_id=20000,
    start_element_id=20000,
    load_configs=None,
    mass_configs=None,
    visualize=True,
    output_dir="output",
    slab_configs=None,
    existing_frame_nodes=None
)
'''
    }
    
    return examples


def create_zip_archive(output_dir):
    """Create ZIP archive of output directory"""
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                z.write(file_path, arcname)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Structural Model Builder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL STYLES
# ============================================================

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

/* Main Header */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    font-weight: 600;
    height: 3.5rem;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    font-size: 1rem;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
}

/* Cards */
.example-card {
    background: white;
    border: 2px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.example-card:hover {
    border-color: #667eea;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    transform: translateY(-4px);
}

.example-card.selected {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
}

.example-title {
    font-weight: 600;
    font-size: 1.2rem;
    color: #1f2937;
    margin-bottom: 0.5rem;
}

.example-desc {
    color: #6b7280;
    font-size: 0.95rem;
}

/* Metrics */
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Section Headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
}

/* Code Editor */
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
}

/* Success/Error Messages */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 12px;
    padding: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
}

/* Image containers */
.stImage {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Expander */
.streamlit-expanderHeader {
    background: #f9fafb;
    border-radius: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if 'examples' not in st.session_state:
    st.session_state.examples = get_default_examples()

if 'config_text' not in st.session_state:
    st.session_state.config_text = ""

if 'config_added' not in st.session_state:
    st.session_state.config_added = False

if 'model_built' not in st.session_state:
    st.session_state.model_built = False

if 'output_dir' not in st.session_state:
    st.session_state.output_dir = "output"

if 'build_results' not in st.session_state:
    st.session_state.build_results = None

if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# ============================================================
# SIDEBAR - EXAMPLE MANAGEMENT
# ============================================================

with st.sidebar:
    st.markdown("### 📚 Example Library")
    
    # Display existing examples
    if st.session_state.examples:
        for key, example in st.session_state.examples.items():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"📄 {example['name']}", 
                        key=f"load_{key}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_example == key else "secondary"
                    ):
                        st.session_state.selected_example = key
                        st.session_state.config_text = example['code']
                        st.session_state.config_added = False
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{key}", help="Delete example"):
                        del st.session_state.examples[key]
                        if st.session_state.selected_example == key:
                            st.session_state.selected_example = None
                            st.session_state.config_text = ""
                        st.rerun()
    else:
        st.info("No examples available. Add one below!")
    
    st.markdown("---")
    
    # Add new example
    st.markdown("### ➕ Add New Example")
    
    with st.form("add_example_form"):
        new_name = st.text_input("Example Name", placeholder="My Custom Model")
        new_desc = st.text_input("Description", placeholder="Brief description...")
        new_code = st.text_area("Code", height=200, placeholder="Paste your Python code here...")
        
        if st.form_submit_button("Add Example", use_container_width=True):
            if new_name and new_code:
                key = new_name.lower().replace(' ', '_')
                st.session_state.examples[key] = {
                    'name': new_name,
                    'description': new_desc or "Custom example",
                    'code': new_code
                }
                save_examples_to_file(st.session_state.examples)
                st.success(f"✅ Added: {new_name}")
                st.rerun()
            else:
                st.error("Name and code are required!")
    
    st.markdown("---")
    
    # Import/Export examples
    st.markdown("### 💾 Import/Export")
    
    # Export
    if st.session_state.examples:
        examples_json = json.dumps(st.session_state.examples, indent=2)
        st.download_button(
            "📥 Export Examples",
            examples_json,
            "examples.json",
            "application/json",
            use_container_width=True
        )
    
    # Import
    uploaded_examples = st.file_uploader("📤 Import Examples", type=['json'])
    if uploaded_examples:
        try:
            imported = json.loads(uploaded_examples.read().decode('utf-8'))
            st.session_state.examples.update(imported)
            save_examples_to_file(st.session_state.examples)
            st.success("✅ Examples imported!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Import failed: {e}")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    new_output_dir = st.text_input("Output Directory", value=st.session_state.output_dir)
    if new_output_dir != st.session_state.output_dir:
        st.session_state.output_dir = new_output_dir

# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown('<h1 class="main-title">🏗️ Structural Model Builder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Finite Element Analysis Platform</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["📝 Editor", "🚀 Build & Results", "📊 Analytics"])

# ============================================================
# TAB 1: EDITOR
# ============================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Configuration Editor</div>', unsafe_allow_html=True)
        
        if st.session_state.config_text:
            if st.session_state.selected_example:
                example_data = st.session_state.examples.get(st.session_state.selected_example)
                if example_data:
                    st.info(f"📝 Editing: **{example_data['name']}** - {example_data['description']}")
            
            edited_config = st.text_area(
                "Edit your configuration:",
                value=st.session_state.config_text,
                height=500,
                key="config_editor"
            )
            
            if edited_config != st.session_state.config_text:
                st.session_state.config_text = edited_config
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("✅ Validate", type="primary", use_container_width=True):
                    is_valid, msg = validate_config(st.session_state.config_text)
                    if is_valid:
                        st.session_state.config_added = True
                        st.success("✅ Configuration is valid!")
                    else:
                        st.error(f"❌ {msg}")
            
            with col_b:
                if st.button("🔄 Reset", use_container_width=True):
                    if st.session_state.selected_example and st.session_state.selected_example in st.session_state.examples:
                        st.session_state.config_text = st.session_state.examples[st.session_state.selected_example]['code']
                        st.rerun()
            
            with col_c:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.config_text = ""
                    st.session_state.selected_example = None
                    st.session_state.config_added = False
                    st.rerun()
        
        else:
            st.info("👈 Select an example from the sidebar to get started!")
            
            # Show code template
            with st.expander("📖 Show Code Template"):
                st.code('''# Basic Template
import numpy as np
from test3 import build_model

# Define your materials, nodes, elements...
# Then call build_model(...)

results = build_model(
    model_params={'ndm': 3, 'ndf': 6},
    # ... other parameters
)
''', language='python')
    
    with col2:
        st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        
        # Upload file
        uploaded_file = st.file_uploader("📤 Upload Python File", type=['py'])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                st.session_state.config_text = content
                st.session_state.selected_example = "custom_upload"
                st.session_state.config_added = False
                st.success("✅ File loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # Save current config
        if st.session_state.config_text:
            st.download_button(
                "💾 Download Configuration",
                st.session_state.config_text.encode('utf-8'),
                f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                "text/x-python",
                use_container_width=True
            )
        
        # Quick guide
        with st.expander("❓ Quick Guide"):
            st.markdown("""
            **Steps:**
            1. Select example from sidebar
            2. Edit configuration as needed
            3. Click 'Validate'
            4. Go to 'Build & Results' tab
            5. Click 'Build Model'
            
            **Requirements:**
            - Import `build_model` from `test3`
            - Define all required parameters
            - Call `build_model(...)`
            """)

# ============================================================
# TAB 2: BUILD & RESULTS
# ============================================================

with tab2:
    if st.session_state.config_added:
        st.markdown('<div class="section-header">Build Model</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with st.expander("📋 Configuration Preview"):
                st.code(st.session_state.config_text, language='python', line_numbers=True)
        
        with col2:
            if st.button("🔨 Build Model", type="primary", use_container_width=True):
                with st.spinner("Building model..."):
                    try:
                        od = st.session_state.output_dir
                        if os.path.exists(od):
                            shutil.rmtree(od)
                        os.makedirs(od, exist_ok=True)
                        
                        patch_gmsh()
                        
                        from test3 import build_model, generate_complete_model_file, create_regular_polygon_nodes
                        
                        eg = {
                            'build_model': build_model,
                            'generate_complete_model_file': generate_complete_model_file,
                            'np': __import__('numpy'),
                            'opst': __import__('opstool'),
                            'create_regular_polygon_nodes': create_regular_polygon_nodes
                        }
                        
                        exec(st.session_state.config_text, eg)
                        
                        if 'results' in eg:
                            st.session_state.build_results = eg['results']
                        
                        st.session_state.model_built = True
                        st.success("✅ Model built successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Build error: {e}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc(), language='bash')
    else:
        st.info("⚠️ Please validate your configuration in the Editor tab first!")
    
    # Results section
    if st.session_state.model_built and os.path.exists(st.session_state.output_dir):
        st.markdown("---")
        st.markdown('<div class="section-header">Build Results</div>', unsafe_allow_html=True)
        
        files = list(Path(st.session_state.output_dir).rglob("*"))
        imgs = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg'] and f.is_file()]
        pys = [f for f in files if f.suffix == '.py' and f.is_file()]
        htmls = [f for f in files if f.suffix == '.html' and f.is_file()]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{st.session_state.build_results.get('total_nodes', 'N/A') if st.session_state.build_results else '✓'}</div>
                <div class="metric-label">Nodes</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{st.session_state.build_results.get('total_elements', 'N/A') if st.session_state.build_results else '✓'}</div>
                <div class="metric-label">Elements</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{len(imgs)}</div>
                <div class="metric-label">Images</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{len(pys) + len(htmls)}</div>
                <div class="metric-label">Files</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Images
        if imgs:
            st.markdown("### 🖼️ Generated Images")
            cols = st.columns(2)
            for i, img in enumerate(sorted(imgs)):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_column_width=True)
                    with open(img, 'rb') as f:
                        st.download_button(
                            f"⬇️ {img.name}",
                            f.read(),
                            img.name,
                            key=f"img_{i}",
                            use_container_width=True
                        )
        
        # 3D Visualizations
        if htmls:
            st.markdown("### 🌐 3D Visualizations")
            for i, h in enumerate(sorted(htmls)):
                with st.expander(f"📈 {h.name}", expanded=(i == 0)):
                    with open(h, encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=600, scrolling=True)
                    
                    with open(h, 'rb') as f:
                        st.download_button(
                            f"⬇️ Download {h.name}",
                            f.read(),
                            h.name,
                            key=f"html_{i}",
                            use_container_width=True
                        )
        
        # Python Files
        if pys:
            st.markdown("### 🐍 Python Files")
            for i, p in enumerate(sorted(pys)):
                with st.expander(f"📄 {p.name}"):
                    with open(p) as f:
                        st.code(f.read(), language='python', line_numbers=True)
                    
                    with open(p, 'rb') as f:
                        st.download_button(
                            f"⬇️ Download {p.name}",
                            f.read(),
                            p.name,
                            key=f"py_{i}",
                            use_container_width=True
                        )
        
        st.markdown("---")
        
        # Download all
        col1, col2 = st.columns(2)
        with col1:
            zip_data = create_zip_archive(st.session_state.output_dir)
            st.download_button(
                "📦 Download All Files (ZIP)",
                zip_data,
                f"model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip",
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Clean Output Directory", use_container_width=True):
                if os.path.exists(st.session_state.output_dir):
                    shutil.rmtree(st.session_state.output_dir)
                st.session_state.model_built = False
                st.session_state.build_results = None
                st.success("✅ Output directory cleaned!")
                st.rerun()

# ============================================================
# TAB 3: ANALYTICS
# ============================================================

with tab3:
    st.markdown('<div class="section-header">Model Statistics</div>', unsafe_allow_html=True)
    
    if st.session_state.build_results:
        st.write("### Build Information")
        st.json(st.session_state.build_results)
    else:
        st.info("📊 Build a model to see analytics here!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <p style='color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;'>
        🏗️ <strong>Structural Model Builder</strong>
    </p>
    <p style='color: #9ca3af; font-size: 0.8rem;'>
        Advanced Finite Element Analysis Platform
    </p>
</div>
""", unsafe_allow_html=True)






