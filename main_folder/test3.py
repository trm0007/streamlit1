"""
Minimal Dynamic GMSH Mesh Generator
Clean implementation with no defaults - all parameters required
"""

import os
import pickle
from math import sqrt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import gmsh

import openseespy.opensees as ops
import opstool as opst
import opstool.vis.pyvista as opsvis
import opstool.vis.plotly as opsvis_plotly


"""
Shell Design Helper Functions
"""

import numpy as np


def create_regular_polygon_nodes(center_x, center_y, radius, n_sides, start_id, z=0.0):
    """
    Create regular polygon nodes dictionary
    
    Parameters
    ----------
    center_x : float
        X coordinate of center
    center_y : float
        Y coordinate of center
    radius : float
        Radius of polygon
    n_sides : int
        Number of sides
    start_id : int
        Starting node ID
    z : float
        Z coordinate (elevation)
    
    Returns
    -------
    dict
        Node dictionary {node_id: (x, y, z)}
    """
    angles = np.linspace(0, 2*np.pi, n_sides + 1)[:-1]
    nodes = {}
    for i, angle in enumerate(angles):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        nodes[start_id + i] = (x, y, z)
    return nodes

def generate_mesh(boundary_nodes, mesh_size, internal_points, voids,
                  py_file, png_file, material_E, material_nu, material_rho,
                  thickness, node_font_size, element_font_size, 
                  start_node_id, start_element_id):
    """Generate mesh with boundary nodes, internal points, and voids"""
    
    gmsh.initialize()
    gmsh.model.add("mesh")
    
    # Sort boundary nodes by angle
    coords = np.array([boundary_nodes[nid] for nid in sorted(boundary_nodes.keys())])
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_ids = [sorted(boundary_nodes.keys())[i] for i in sorted_indices]
    
    # Create boundary
    point_map = {}
    for node_id in sorted_ids:
        x, y, z = boundary_nodes[node_id]
        pt = gmsh.model.geo.addPoint(x, y, z, mesh_size)
        point_map[node_id] = pt
    
    boundary_lines = []
    for i in range(len(sorted_ids)):
        start = sorted_ids[i]
        end = sorted_ids[(i + 1) % len(sorted_ids)]
        line = gmsh.model.geo.addLine(point_map[start], point_map[end])
        boundary_lines.append(line)
    
    outer_loop = gmsh.model.geo.addCurveLoop(boundary_lines)
    
    # Process voids
    void_loops = []
    if voids:
        for void_nodes in voids:
            void_coords = np.array([void_nodes[nid] for nid in sorted(void_nodes.keys())])
            void_center = void_coords.mean(axis=0)
            void_angles = np.arctan2(void_coords[:, 1] - void_center[1], 
                                     void_coords[:, 0] - void_center[0])
            void_sorted_indices = np.argsort(void_angles)
            void_sorted_ids = [sorted(void_nodes.keys())[i] for i in void_sorted_indices]
            
            void_point_map = {}
            for node_id in void_sorted_ids:
                x, y, z = void_nodes[node_id]
                pt = gmsh.model.geo.addPoint(x, y, z, mesh_size)
                void_point_map[node_id] = pt
            
            void_lines = []
            for i in range(len(void_sorted_ids)):
                start = void_sorted_ids[i]
                end = void_sorted_ids[(i + 1) % len(void_sorted_ids)]
                line = gmsh.model.geo.addLine(void_point_map[start], void_point_map[end])
                void_lines.append(line)
            
            void_loop = gmsh.model.geo.addCurveLoop(void_lines)
            void_loops.append(void_loop)
    
    all_loops = [outer_loop] + void_loops
    surface = gmsh.model.geo.addPlaneSurface(all_loops)
    gmsh.model.geo.synchronize()
    
    # Embed internal points
    if internal_points:
        for node_id, coord in internal_points.items():
            x, y, z = coord
            pt = gmsh.model.geo.addPoint(x, y, z, mesh_size)
            gmsh.model.geo.synchronize()
            try:
                gmsh.model.mesh.embed(0, [pt], 2, surface)
            except:
                pass
    
    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2)
    
    try:
        gmsh.model.mesh.generate(2)
    except:
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.model.mesh.generate(2)
    
    # Extract mesh
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    temp_nodes = {}
    for i, tag in enumerate(node_tags):
        temp_nodes[int(tag)] = (node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2])

    quad4_elems = []
    tri3_elems = []

    for elem_type in gmsh.model.mesh.getElementTypes(dim=2):
        elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(elem_type)
        
        if elem_type == 3:
            for i, tag in enumerate(elem_tags):
                nodes = [int(elem_nodes[i*4 + j]) for j in range(4)]
                quad4_elems.append({'tag': int(tag), 'nodes': nodes})
        elif elem_type == 2:
            for i, tag in enumerate(elem_tags):
                nodes = [int(elem_nodes[i*3 + j]) for j in range(3)]
                tri3_elems.append({'tag': int(tag), 'nodes': nodes})
    
    gmsh.finalize()
    
    # Node mapping - preserve special IDs
    tolerance = mesh_size * 0.01
    node_map = {}
    final_nodes = {}
    used_ids = set()
    
    # Match boundary nodes
    for bnd_id, bnd_coord in boundary_nodes.items():
        best = None
        best_dist = float('inf')
        for gid, gcoord in temp_nodes.items():
            if gid in node_map:
                continue
            dist = np.linalg.norm(np.array(gcoord) - np.array(bnd_coord))
            if dist < tolerance and dist < best_dist:
                best = gid
                best_dist = dist
        if best:
            node_map[best] = bnd_id
            final_nodes[bnd_id] = bnd_coord
            used_ids.add(bnd_id)
    
    # Match void nodes
    if voids:
        for void_nodes in voids:
            for void_id, void_coord in void_nodes.items():
                best = None
                best_dist = float('inf')
                for gid, gcoord in temp_nodes.items():
                    if gid in node_map:
                        continue
                    dist = np.linalg.norm(np.array(gcoord) - np.array(void_coord))
                    if dist < tolerance and dist < best_dist:
                        best = gid
                        best_dist = dist
                if best:
                    node_map[best] = void_id
                    final_nodes[void_id] = void_coord
                    used_ids.add(void_id)
    
    # Match internal points - keep original IDs
    matched_internal = {}
    if internal_points:
        for int_id, int_coord in internal_points.items():
            best = None
            best_dist = float('inf')
            for gid, gcoord in temp_nodes.items():
                if gid in node_map:
                    continue
                dist = np.linalg.norm(np.array(gcoord) - np.array(int_coord))
                if dist < tolerance and dist < best_dist:
                    best = gid
                    best_dist = dist
            
            if best:
                node_map[best] = int_id
                final_nodes[int_id] = int_coord
                used_ids.add(int_id)
                matched_internal[int_id] = int_id
    
    # Sequential numbering for remaining
    remaining = sorted([g for g in temp_nodes.keys() if g not in node_map])
    next_id = start_node_id
    for gid in remaining:
        while next_id in used_ids:
            next_id += 1
        node_map[gid] = next_id
        final_nodes[next_id] = temp_nodes[gid]
        used_ids.add(next_id)
        next_id += 1
    
    # Remap elements
    elem_id = start_element_id
    final_quad4 = []
    final_tri3 = []
    
    for elem in quad4_elems:
        nodes = [node_map[n] for n in elem['nodes']]
        final_quad4.append({'tag': elem_id, 'nodes': nodes})
        elem_id += 1
    
    for elem in tri3_elems:
        nodes = [node_map[n] for n in elem['nodes']]
        final_tri3.append({'tag': elem_id, 'nodes': nodes})
        elem_id += 1
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 12))
    
    for elem in final_quad4:
        coords = np.array([final_nodes[n][:2] for n in elem['nodes']])
        ax.fill(coords[:, 0], coords[:, 1], fc='cyan', ec='blue', alpha=0.3, lw=1)
    for elem in final_tri3:
        coords = np.array([final_nodes[n][:2] for n in elem['nodes']])
        ax.fill(coords[:, 0], coords[:, 1], fc='yellow', ec='orange', alpha=0.3, lw=1)
    
    bnd_ids = set(boundary_nodes.keys())
    void_ids = set()
    if voids:
        for v in voids:
            void_ids.update(v.keys())
    int_ids = set(matched_internal.keys())
    reg_ids = set(final_nodes.keys()) - bnd_ids - void_ids - int_ids
    
    if reg_ids:
        coords = np.array([final_nodes[n][:2] for n in reg_ids])
        ax.scatter(coords[:, 0], coords[:, 1], c='black', s=30, zorder=5)
    
    coords = np.array([boundary_nodes[n][:2] for n in boundary_nodes])
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=150, marker='s', 
               ec='black', lw=2, label='Boundary', zorder=6)
    
    if voids:
        for void_nodes in voids:
            coords = np.array([void_nodes[n][:2] for n in void_nodes])
            ax.scatter(coords[:, 0], coords[:, 1], c='purple', s=120, marker='o',
                       ec='black', lw=2, zorder=6)
    
    if int_ids:
        coords = np.array([final_nodes[n][:2] for n in int_ids])
        ax.scatter(coords[:, 0], coords[:, 1], c='lime', s=300, marker='^',
                   ec='darkgreen', lw=3, label='Internal', zorder=8)
        
        for nid in int_ids:
            coord = final_nodes[nid]
            ax.annotate(str(nid), (coord[0], coord[1]), xytext=(0, 20),
                       textcoords='offset points', fontsize=node_font_size+3,
                       ha='center', color='darkgreen', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', fc='lime', ec='darkgreen', lw=2.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='darkgreen'),
                       zorder=11)
    
    for nid in bnd_ids:
        if nid in final_nodes:
            coord = final_nodes[nid]
            ax.annotate(str(nid), (coord[0], coord[1]), xytext=(0, -15),
                       textcoords='offset points', fontsize=node_font_size,
                       ha='center', color='darkred', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', ec='darkred'),
                       zorder=9)
    
    for nid in reg_ids:
        coord = final_nodes[nid]
        ax.annotate(str(nid), (coord[0], coord[1]), fontsize=node_font_size-1,
                   ha='center', color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6),
                   zorder=7)
    
    title = f'{len(final_nodes)} nodes, {len(final_quad4)} quad4, {len(final_tri3)} tri3'
    if int_ids:
        title += f', {len(int_ids)} internal'
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_file, dpi=200)
    plt.close()
    
    print(f"Mesh: {len(final_nodes)} nodes, {len(final_quad4)} quad4, {len(final_tri3)} tri3")
    
    return {
        'nodes': final_nodes,
        'quad4': final_quad4,
        'tri3': final_tri3,
        'voids': voids if voids else [],
        'internal_points': matched_internal
    }


def zero_element_boundary_condition(material_props, sections, node_list, 
                                    boundary_condition, element_start_id, 
                                    spring_node_start_id):
    """Create zero-length elements at specified nodes"""
    
    node_mapping = {}
    element_ids = []
    current_elem_id = element_start_id
    current_spring_node_id = spring_node_start_id
    
    try:
        ops.uniaxialMaterial(*material_props['config'])
    except:
        pass
    
    for node_id, x, y, z in node_list:
        spring_node_id = current_spring_node_id
        
        ops.node(spring_node_id, x, y, z)
        ops.fix(spring_node_id, *boundary_condition)
        
        ops.element("zeroLength", current_elem_id, 
                   node_id, spring_node_id, 
                   "-mat", material_props['id'], 
                   "-dir", *material_props['directions'])
        
        node_mapping[node_id] = {
            'spring_node': spring_node_id,
            'element_id': current_elem_id,
            'main_coords': (x, y, z),
            'spring_coords': (x, y, z)
        }
        
        element_ids.append(current_elem_id)
        current_elem_id += 1
        current_spring_node_id += 1
    
    return {
        'node_mapping': node_mapping,
        'element_ids': element_ids,
        'spring_node_ids': list(range(spring_node_start_id, current_spring_node_id)),
        'total_elements': len(element_ids)
    }


def create_slab(boundary_nodes, mesh_size, internal_points, voids,
                py_file, png_file, shell_material_config, shell_section_config,
                node_font_size, element_font_size, ops_ele_type1, ops_ele_type2,
                shell_boundary_conditions, use_zero_length, 
                zero_length_material_config, zero_length_directions,
                zero_length_boundary_conditions, element_start_id,
                spring_node_start_id, load_configs, start_node_id, 
                start_element_id):
    """Create slab with mesh generation"""
    
    material_E = shell_material_config[2]
    material_nu = shell_material_config[3]
    material_rho = shell_material_config[4]
    thickness = shell_section_config[3]
    
    mesh = generate_mesh(
        boundary_nodes=boundary_nodes,
        mesh_size=mesh_size,
        internal_points=internal_points,
        voids=voids,
        py_file=py_file,
        png_file=png_file,
        material_E=material_E,
        material_nu=material_nu,
        material_rho=material_rho,
        thickness=thickness,
        node_font_size=node_font_size,
        element_font_size=element_font_size,
        start_node_id=start_node_id,
        start_element_id=start_element_id
    )
    
    return mesh


def create_dynamic_composite_section(
    materials, outline_points, core_material, mesh_sizes, ops_mat_tags,
    cover_thickness, cover_material, core_holes, voids, bone_geometry,
    additional_patches, rebar_configs, steel_material, sec_tag,
    save_txt_path, save_png_path, save_pkl_path, G, section_name,
    display_results, plot_section):
    """Create dynamic composite section - all parameters required"""
    
    for name, props in materials.items():
        if 'elastic_modulus' not in props:
            raise ValueError(f"Material '{name}': 'elastic_modulus' required")
        if 'poissons_ratio' not in props:
            raise ValueError(f"Material '{name}': 'poissons_ratio' required")
        if 'density' not in props:
            raise ValueError(f"Material '{name}': 'density' required")
    
    mat_objects = {}
    for name, props in materials.items():
        mat_objects[name] = opst.pre.section.create_material(
            name=name,
            elastic_modulus=props['elastic_modulus'],
            poissons_ratio=props['poissons_ratio'],
            density=props['density'],
            yield_strength=props.get('yield_strength', 1.0),
            color=props.get('color', 'gray')
        )
    
    all_voids = []
    if voids:
        for v in voids:
            if v['type'] == 'polygon':
                all_voids.append(v['points'])
            elif v['type'] == 'circle':
                pts = opst.pre.section.create_circle_points(
                    v['xo'], v['radius'], 
                    v.get('angles', (0, 360)), 
                    v.get('n_sub', 40)
                )
                all_voids.append(pts)
    
    combined_holes = (core_holes or []) + all_voids
    
    patches = {}
    
    if cover_thickness and cover_material:
        coverlines = opst.pre.section.offset(outline_points, d=cover_thickness)
        patches['cover'] = opst.pre.section.create_polygon_patch(
            outline_points, 
            holes=[coverlines], 
            material=mat_objects[cover_material]
        )
    else:
        coverlines = outline_points
    
    if not cover_thickness:
        patches['core'] = opst.pre.section.create_polygon_patch(
            outline_points, 
            holes=combined_holes or None, 
            material=mat_objects[core_material]
        )
    else:
        patches['core'] = opst.pre.section.create_polygon_patch(
            coverlines, 
            holes=combined_holes or None, 
            material=mat_objects[core_material]
        )
    
    if bone_geometry:
        patches['bone'] = opst.pre.section.create_polygon_patch(
            bone_geometry['points'], 
            holes=bone_geometry.get('holes'),
            material=mat_objects[bone_geometry['material']]
        )
    
    if additional_patches:
        for p in additional_patches:
            patches[p['name']] = opst.pre.section.create_polygon_patch(
                p['points'], 
                holes=p.get('holes'), 
                material=mat_objects[p['material']]
            )
    
    SEC = opst.pre.section.FiberSecMesh(sec_name=section_name)
    SEC.add_patch_group(patches)
    SEC.set_mesh_size(mesh_sizes)
    
    color_map = {}
    if 'cover' in patches:
        color_map['cover'] = materials.get(cover_material, {}).get('color', '#dbb40c')
    if 'core' in patches:
        color_map['core'] = materials.get(core_material, {}).get('color', '#88b378')
    if 'bone' in patches:
        color_map['bone'] = materials.get(bone_geometry['material'], {}).get('color', '#ffc168')
    for p in (additional_patches or []):
        color_map[p['name']] = materials.get(p['material'], {}).get('color', 'gray')
    SEC.set_mesh_color(color_map)
    
    patch_mat_tags = {k: v for k, v in ops_mat_tags.items() if k in patches}
    SEC.set_ops_mat_tag(patch_mat_tags)
    SEC.mesh()
    
    if rebar_configs and steel_material:
        rebar_tag = ops_mat_tags.get('rebar')
        for i, cfg in enumerate(rebar_configs):
            t = cfg.get('type', 'line')
            if t == 'line':
                SEC.add_rebar_line(
                    cfg['points'], cfg['dia'], 
                    cfg.get('gap', 0.1), 
                    cfg.get('n'),
                    cfg.get('closure', False), 
                    rebar_tag, 
                    cfg.get('color', 'black'),
                    cfg.get('group_name', f'Rebar_{i+1}')
                )
            elif t == 'circle':
                SEC.add_rebar_circle(
                    cfg['xo'], cfg['radius'], cfg['dia'], 
                    cfg.get('gap', 0.1),
                    cfg.get('n'), 
                    cfg.get('angles', (0, 360)), 
                    rebar_tag,
                    cfg.get('color', 'black'), 
                    cfg.get('group_name', f'Rebar_{i+1}')
                )
            elif t == 'points':
                SEC.add_rebar_points(
                    cfg['points'], cfg['dia'], rebar_tag,
                    cfg.get('color', 'black'), 
                    cfg.get('group_name', f'Rebar_{i+1}')
                )
    
    SEC.centring()
    
    if display_results:
        SEC.get_frame_props(display_results=True)
    
    if save_txt_path and sec_tag is not None:
        if G is None:
            raise ValueError("G value required")
        GJ = G * SEC.get_j()
        SEC.to_file(save_txt_path, secTag=sec_tag, GJ=GJ, fmt=":.6E")
        
        if save_txt_path.endswith('.py'):
            params_file = save_txt_path.replace('.py', '_params.txt')
        else:
            params_file = save_txt_path + '_params.txt'
            
        with open(params_file, 'w') as f:
            f.write(f"section_tag = {sec_tag}\n")
            f.write(f"GJ = {GJ:.6E}\n")
            f.write(f"section_name = '{section_name}'\n")
    
    if save_png_path:
        fig, ax = plt.subplots(figsize=(8, 8))
        SEC.view(fill=True, show_legend=True, ax=ax)
        ax.set_aspect("equal", "box")
        plt.tight_layout()
        plt.savefig(save_png_path, dpi=300, bbox_inches='tight')
        if not plot_section:
            plt.close(fig)
    
    if save_pkl_path:
        with open(save_pkl_path, 'wb') as f:
            pickle.dump(SEC, f)
    
    if plot_section and not save_png_path:
        SEC.view(fill=True, show_legend=True)
        plt.show()
    
    return SEC, sec_tag, save_txt_path, save_png_path, save_pkl_path, params_file


def load_saved_section(txt_path, png_path, pkl_path, display_commands, 
                      display_image, return_section_object):
    """Load saved section files"""
    
    commands, fig, section = None, None, None
    section_id = None
    GJ_value = None
    file_paths = {'txt': txt_path, 'png': png_path, 'pkl': pkl_path}
    
    if txt_path:
        if txt_path.endswith('.py'):
            params_file = txt_path.replace('.py', '_params.txt')
        else:
            params_file = txt_path + '_params.txt'
        
        try:
            with open(params_file, 'r') as f:
                for line in f:
                    if line.startswith('section_tag'):
                        section_id = int(line.split('=')[1].strip())
                    elif line.startswith('GJ'):
                        GJ_value = float(line.split('=')[1].strip())
        except:
            pass
    
    if txt_path:
        try:
            with open(txt_path, 'r') as f:
                commands = f.read()
            if display_commands:
                print(commands)
        except Exception as e:
            print(f"Error loading {txt_path}: {e}")
    
    if png_path and display_image:
        try:
            img = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Loaded from: {png_path}', fontsize=12)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error loading {png_path}: {e}")
    
    if pkl_path and return_section_object:
        try:
            with open(pkl_path, 'rb') as f:
                section = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
    
    return commands, fig, section, section_id, GJ_value, file_paths


def create_fiber_section(materials, outline_points, cover, rebar_configs, mesh_size, 
                         mat_tags, sec_tag, G, save_prefix, section_name):
    """Create and save fiber section"""
    SEC, sec_id, txt_path, png_path, pkl_path, params_path = create_dynamic_composite_section(
        materials=materials,
        outline_points=outline_points,
        cover_thickness=cover,
        cover_material='concrete_cover',
        core_material='concrete_core',
        mesh_sizes={'cover': mesh_size, 'core': mesh_size},
        ops_mat_tags=mat_tags,
        rebar_configs=rebar_configs,
        steel_material='steel_rebar',
        sec_tag=sec_tag,
        G=G,
        save_txt_path=f'{save_prefix}_commands.py',
        save_png_path=f'{save_prefix}_figure.png',
        save_pkl_path=f'{save_prefix}_object.pkl',
        section_name=section_name,
        display_results=False,
        plot_section=False,
        core_holes=None,
        voids=None,
        bone_geometry=None,
        additional_patches=None
    )
    return sec_id, txt_path, png_path, pkl_path


def define_uniaxial_materials(material_params):
    """Define OpenSees uniaxial materials"""
    for param in material_params:
        ops.uniaxialMaterial(*param)


def create_nodes(node_coords):
    """Create nodes"""
    for node_id, coords in node_coords.items():
        ops.node(node_id, *coords)


def apply_boundary_conditions(boundary_conditions):
    """Apply boundary conditions"""
    for node_id, dofs in boundary_conditions.items():
        ops.fix(node_id, *dofs)


def create_rigid_diaphragms(diaphragm_list):
    """Create rigid diaphragm constraints"""
    for perp_dir, ret_node, *constr_nodes in diaphragm_list:
        ops.rigidDiaphragm(perp_dir, ret_node, *constr_nodes)


def create_elements(element_configs):
    """Create transformations, integrations, elastic sections, and elements"""
    
    for transf in element_configs.get('transformations', []):
        ops.geomTransf(transf['type'], transf['tag'], *transf['vecxz'])
    
    for integ in element_configs.get('integrations', []):
        ops.beamIntegration(integ['type'], integ['tag'], integ['sec_tag'], integ['np'])
    
    for elastic_sec in element_configs.get('elastic_sections', []):
        ops.section("Elastic", elastic_sec['sec_tag'], elastic_sec['E'],
                   elastic_sec['A'], elastic_sec['Iz'], elastic_sec['Iy'],
                   elastic_sec['G'], elastic_sec['J'])
    
    for col in element_configs.get('force_beam_columns', []):
        ops.element("forceBeamColumn", col['tag'], col['node_i'], col['node_j'], 
                    col['transf_tag'], col['integ_tag'])
    
    for beam in element_configs.get('elastic_beam_columns', []):
        ops.element("elasticBeamColumn", beam['tag'], beam['node_i'], beam['node_j'],
                    beam['A'], beam['E'], beam['G'], beam['J'], 
                    beam['Iy'], beam['Iz'], beam['transf_tag'])


def apply_loads_and_masses(load_configs, mass_configs, shell_meshes, 
                           slab_configs, element_configs, node_coords):
    """Apply all loads and masses - all parameters required"""
    
    if load_configs is None and mass_configs is None:
        raise ValueError("Both load_configs and mass_configs are None")
    
    if element_configs is None:
        raise ValueError("element_configs required")
    
    if node_coords is None:
        raise ValueError("node_coords required")
    
    try:
        import opstool as opst
        has_opstool = True
    except ImportError:
        opst = None
        has_opstool = False
    
    results = {
        'nodal_masses': {},
        'load_summary': {},
        'mass_summary': {}
    }
    
    # Apply loads
    if load_configs is not None:
        print("\nApplying loads...")
        
        if 'time_series' in load_configs:
            for ts in load_configs['time_series']:
                tag = ts['tag']
                ts_type = ts['type']
                
                if ts_type == 'Linear':
                    ops.timeSeries('Linear', tag)
                elif ts_type == 'Constant':
                    ops.timeSeries('Constant', tag)
                elif ts_type == 'Trig':
                    ops.timeSeries('Trig', tag, ts['tStart'], ts['tEnd'], ts['period'])
            
            results['load_summary']['time_series'] = len(load_configs['time_series'])
        
        if 'patterns' in load_configs:
            for pattern in load_configs['patterns']:
                ops.pattern('Plain', pattern['tag'], pattern['ts_tag'])
            
            results['load_summary']['patterns'] = len(load_configs['patterns'])
        
        if 'nodal_loads' in load_configs:
            total_nodal_loads = 0
            for load_group in load_configs['nodal_loads']:
                for load in load_group['loads']:
                    ops.load(load['node'], *load['forces'])
                    total_nodal_loads += 1
            
            results['load_summary']['nodal_loads'] = total_nodal_loads
        
        if 'beam_uniform_loads' in load_configs:
            if not has_opstool:
                raise ImportError("opstool required for beam_uniform_loads")
            
            total_beam_uniform = 0
            for load_group in load_configs['beam_uniform_loads']:
                for load in load_group['loads']:
                    opst.pre.transform_beam_uniform_load(load['elements'], 
                                                        wy=load['wy'], 
                                                        wz=load['wz'])
                    total_beam_uniform += len(load['elements'])
            
            results['load_summary']['beam_uniform_loads'] = total_beam_uniform
        
        if 'beam_point_loads' in load_configs:
            if not has_opstool:
                raise ImportError("opstool required for beam_point_loads")
            
            total_beam_point = 0
            for load_group in load_configs['beam_point_loads']:
                for load in load_group['loads']:
                    opst.pre.transform_beam_point_load([load['element']], 
                                                       py=load['py'], 
                                                       pz=load['pz'], 
                                                       xl=load['xl'])
                    total_beam_point += 1
            
            results['load_summary']['beam_point_loads'] = total_beam_point
        
        if 'shell_surface_loads' in load_configs:
            if not has_opstool:
                raise ImportError("opstool required for shell_surface_loads")
            
            if not shell_meshes:
                raise ValueError("shell_meshes required for shell_surface_loads")
            
            total_shell_loads = 0
            for load_group in load_configs['shell_surface_loads']:
                for load in load_group['loads']:
                    mesh_name = load['mesh_name']
                    pressure = load['pressure']
                    specific_elements = load['elements']
                    
                    target_mesh = None
                    for mesh in shell_meshes:
                        if mesh.get('config_name') == mesh_name:
                            target_mesh = mesh
                            break
                    
                    if target_mesh is None:
                        raise ValueError(f"Mesh '{mesh_name}' not found")
                    
                    if specific_elements is None:
                        element_tags = [elem['tag'] for elem in target_mesh['quad4']]
                        element_tags += [elem['tag'] for elem in target_mesh['tri3']]
                    else:
                        element_tags = specific_elements
                    
                    opst.pre.transform_surface_uniform_load(ele_tags=element_tags, p=pressure)
                    total_shell_loads += len(element_tags)
            
            results['load_summary']['shell_surface_loads'] = total_shell_loads
        
        print("Loads applied")
    
    # Apply masses
    if mass_configs is not None:
        print("\nApplying masses...")
        
        nodal_masses = {node_id: 0.0 for node_id in node_coords.keys()}
        
        if shell_meshes:
            for shell_mesh in shell_meshes:
                for node_id in shell_mesh['nodes'].keys():
                    if node_id not in nodal_masses:
                        nodal_masses[node_id] = 0.0
        
        if 'beam_column_mass' in mass_configs:
            beam_col_mass_applied = 0
            
            for item in mass_configs['beam_column_mass']:
                tag = item['tag']
                density = item['density']
                area = item['area']
                
                element_found = False
                node_i, node_j = None, None
                
                for col in element_configs['force_beam_columns']:
                    if col['tag'] == tag:
                        node_i, node_j = col['node_i'], col['node_j']
                        element_found = True
                        break
                
                if not element_found:
                    for beam in element_configs['elastic_beam_columns']:
                        if beam['tag'] == tag:
                            node_i, node_j = beam['node_i'], beam['node_j']
                            element_found = True
                            break
                
                if not element_found:
                    raise ValueError(f"Element {tag} not found")
                
                xi, yi, zi = node_coords[node_i]
                xj, yj, zj = node_coords[node_j]
                length = ((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)**0.5
                
                mass = density * area * length
                half_mass = mass / 2.0
                
                if node_i not in nodal_masses:
                    nodal_masses[node_i] = 0.0
                if node_j not in nodal_masses:
                    nodal_masses[node_j] = 0.0
                
                nodal_masses[node_i] += half_mass
                nodal_masses[node_j] += half_mass
                
                beam_col_mass_applied += mass
            
            results['mass_summary']['beam_column_mass'] = beam_col_mass_applied
        
        if 'nodal_mass' in mass_configs:
            node_mass_groups = defaultdict(list)
            for item in mass_configs['nodal_mass']:
                node_mass_groups[item['node']].append(item['mass'])
            
            total_nodal_mass_applied = 0.0
            
            for node_id, mass_list in node_mass_groups.items():
                total_mass = sum(mass_list)
                
                if node_id not in nodal_masses:
                    nodal_masses[node_id] = 0.0
                
                nodal_masses[node_id] += total_mass
                total_nodal_mass_applied += total_mass
            
            results['mass_summary']['nodal_mass_total'] = total_nodal_mass_applied
        
        if 'shell_mass' in mass_configs:
            shell_config = mass_configs['shell_mass']
            
            if shell_config['calculate']:
                if not has_opstool:
                    raise ImportError("opstool required for shell mass calculation")
                
                if not shell_meshes:
                    raise ValueError("shell_meshes required when calculate=True")
                
                if not slab_configs:
                    raise ValueError("slab_configs required when calculate=True")
                
                exclude_list = shell_config['exclude']
                scale_factor = shell_config['scale']
                
                total_shell_mass_applied = 0.0
                
                for shell_mesh in shell_meshes:
                    config_name = shell_mesh.get('config_name', 'Unknown')
                    
                    if config_name in exclude_list:
                        continue
                    
                    density = None
                    thickness = None
                    
                    for cfg in slab_configs:
                        if cfg.get('name') == config_name:
                            mat_config = cfg['shell_material_config']
                            density = mat_config[4] * scale_factor
                            
                            sec_config = cfg['shell_section_config']
                            thickness = sec_config[3]
                            break
                    
                    if density is None or thickness is None:
                        raise ValueError(f"Could not find density/thickness for {config_name}")
                    
                    shell_ele_tags = [elem['tag'] for elem in shell_mesh['quad4'] + shell_mesh['tri3']]
                    
                    shell_nodal_masses = _calculate_shell_mass_from_areas(
                        ele_tags=shell_ele_tags,
                        density=density,
                        thickness=thickness,
                        opst=opst
                    )
                    
                    mesh_total_mass = 0.0
                    for node_id, shell_mass in shell_nodal_masses.items():
                        if node_id not in nodal_masses:
                            nodal_masses[node_id] = 0.0
                        
                        nodal_masses[node_id] += shell_mass
                        mesh_total_mass += shell_mass
                    
                    total_shell_mass_applied += mesh_total_mass
                
                results['mass_summary']['shell_mass_total'] = total_shell_mass_applied
        
        nodes_with_mass = 0
        total_mass_applied = 0.0
        
        for node_id, mass_value in nodal_masses.items():
            if mass_value > 0:
                ops.mass(node_id, mass_value, mass_value, mass_value, 0.0, 0.0, 0.0)
                nodes_with_mass += 1
                total_mass_applied += mass_value
        
        results['nodal_masses'] = nodal_masses
        results['mass_summary']['nodes_with_mass'] = nodes_with_mass
        results['mass_summary']['total_mass'] = total_mass_applied
        
        print("Masses applied")
    
    return results


def _compute_tri_area_and_normal(vertices):
    """Compute area and normal of triangle"""
    edge_ij = vertices[1] - vertices[0]
    edge_jk = vertices[2] - vertices[1]
    cross_product = np.cross(edge_ij, edge_jk)
    norm = np.linalg.norm(cross_product)
    area = 0.5 * norm
    normal = cross_product / norm
    return area, normal


def _compute_quad_area_and_normal(vertices):
    """Compute area and normal of quadrilateral"""
    triangle1 = vertices[:3]
    triangle2 = np.array([vertices[0], vertices[2], vertices[3]])
    area1, normal1 = _compute_tri_area_and_normal(triangle1)
    area2, normal2 = _compute_tri_area_and_normal(triangle2)
    normal = (normal1 + normal2) / 2.0
    return area1 + area2, normal


def _calculate_shell_mass_from_areas(ele_tags, density, thickness, opst):
    """Calculate shell element mass from areas"""
    
    ele_tags = [int(tag) for tag in ele_tags]
    nodal_masses = defaultdict(float)
    
    for etag in ele_tags:
        node_ids = ops.eleNodes(etag)
        vertices = np.array([ops.nodeCoord(node_id) for node_id in node_ids])
        
        if len(node_ids) == 3:
            area, _ = _compute_tri_area_and_normal(vertices)
        elif len(node_ids) == 4:
            area, _ = _compute_quad_area_and_normal(vertices)
        else:
            raise ValueError(f"Unsupported element with {len(node_ids)} nodes")
        
        element_mass = density * area * thickness
        mass_per_node = element_mass / len(node_ids)
        
        for node_id in node_ids:
            nodal_masses[node_id] += mass_per_node
    
    return dict(nodal_masses)


def build_model(model_params, materials_list, outline_points_list, 
                rebar_configs_list, section_params_list, material_params,
                node_coords, boundary_conditions, element_configs,
                spring_configs, nodal_spring_configs, start_base_node_id,
                diaphragm_list, start_node_id, start_element_id,
                load_configs, mass_configs, visualize, output_dir,
                slab_configs, existing_frame_nodes):
    """Build complete OpenSeesPy 3D frame model"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)
    
    # Create fiber sections
    print("\nCreating fiber sections...")
    fiber_section_info = []
    for i, (materials, outline_points, rebar_configs, section_params) in enumerate(
        zip(materials_list, outline_points_list, rebar_configs_list, section_params_list)):
        
        sec_id, txt_path, png_path, pkl_path = create_fiber_section(
            materials=materials,
            outline_points=outline_points,
            cover=section_params['cover'],
            rebar_configs=rebar_configs,
            mesh_size=section_params['mesh_size'],
            mat_tags=section_params['mat_tags'],
            sec_tag=section_params['sec_tag'],
            G=section_params['G'],
            save_prefix=os.path.join(output_dir, section_params['save_prefix']),
            section_name=section_params['section_name']
        )
        
        fiber_section_info.append({
            'sec_tag': section_params['sec_tag'],
            'txt_path': txt_path,
            'png_path': png_path,
            'pkl_path': pkl_path,
            'GJ': section_params['G']
        })
    print(f"Created {len(fiber_section_info)} fiber sections")
    
    # Initialize model
    print("\nInitializing OpenSees model...")
    ops.wipe()
    ops.model("basic", "-ndm", model_params['ndm'], "-ndf", model_params['ndf'])
    
    for mat_param in material_params:
        ops.uniaxialMaterial(*mat_param)
    print("Materials defined")
    
    # Create shell meshes
    shell_results = []
    if slab_configs:
        print("\nCreating shell meshes...")
        for i, config in enumerate(slab_configs, start=1):
            config_name = config.get('name', f'Shell_{i}')
            
            mesh = create_slab(
                boundary_nodes=config['boundary_nodes'],
                mesh_size=config['mesh_size'],
                internal_points=config['internal_points'],
                voids=config['voids'],
                py_file=os.path.join(output_dir, config['py_file']),
                png_file=os.path.join(output_dir, config['png_file']),
                shell_material_config=config['shell_material_config'],
                shell_section_config=config['shell_section_config'],
                node_font_size=config['node_font_size'],
                element_font_size=config['element_font_size'],
                ops_ele_type1=config['ops_ele_type1'],
                ops_ele_type2=config['ops_ele_type2'],
                shell_boundary_conditions=config['shell_boundary_conditions'],
                use_zero_length=config['use_zero_length'],
                zero_length_material_config=config['zero_length_material_config'],
                zero_length_directions=config['zero_length_directions'],
                zero_length_boundary_conditions=config['zero_length_boundary_conditions'],
                element_start_id=config['element_start_id'],
                spring_node_start_id=config['spring_node_start_id'],
                load_configs=config['load_configs'],
                start_node_id=config['start_node_id'],
                start_element_id=config['start_element_id']
            )
            
            mesh['config_name'] = config_name
            shell_results.append(mesh)
        
        print(f"Created {len(shell_results)} shell meshes")
    
    # Create nodes
    print("\nCreating nodes...")
    all_nodes_created = set()
    
    for node_id, coords in node_coords.items():
        ops.node(node_id, *coords)
        all_nodes_created.add(node_id)
    
    if shell_results:
        for shell_mesh in shell_results:
            for nid, coords in shell_mesh['nodes'].items():
                if nid not in all_nodes_created:
                    ops.node(nid, coords[0], coords[1], coords[2])
                    all_nodes_created.add(nid)
    
    print(f"Created {len(all_nodes_created)} nodes")
    
    # Apply boundary conditions
    print("\nApplying boundary conditions...")
    for node_id, dofs in boundary_conditions.items():
        ops.fix(node_id, *dofs)
    print(f"Applied to {len(boundary_conditions)} nodes")
    
    # Create fiber sections in OpenSees
    print("\nCreating fiber sections in model...")
    for fiber_sec in fiber_section_info:
        commands, figure, loaded_section, loaded_sec_id, loaded_GJ, file_paths = load_saved_section(
            txt_path=fiber_sec['txt_path'],
            png_path=fiber_sec['png_path'],
            pkl_path=fiber_sec['pkl_path'],
            display_commands=False,
            display_image=False,
            return_section_object=True
        )
        
        if loaded_GJ is not None:
            fiber_sec['GJ'] = loaded_GJ
        
        exec(commands)
    print("Fiber sections created")
    
    # Create springs
    if nodal_spring_configs:
        print("\nCreating support springs...")
        zero_element_boundary_condition(
            material_props=nodal_spring_configs['material_props'],
            sections=nodal_spring_configs.get('sections', {}),
            node_list=nodal_spring_configs['node_list'],
            boundary_condition=nodal_spring_configs['boundary_condition'],
            element_start_id=nodal_spring_configs['element_start_id'],
            spring_node_start_id=nodal_spring_configs['spring_node_start_id']
        )
        print("Springs created")
    
    # Create rigid diaphragms
    if diaphragm_list:
        print("\nCreating rigid diaphragms...")
        for perp_dir, ret_node, *constr_nodes in diaphragm_list:
            ops.rigidDiaphragm(perp_dir, ret_node, *constr_nodes)
        print(f"Created {len(diaphragm_list)} diaphragms")
    
    # Create beam elements
    print("\nCreating beam elements...")
    for transf in element_configs.get('transformations', []):
        ops.geomTransf(transf['type'], transf['tag'], *transf['vecxz'])
    
    for integ in element_configs.get('integrations', []):
        ops.beamIntegration(integ['type'], integ['tag'], integ['sec_tag'], integ['np'])
    
    for elastic_sec in element_configs.get('elastic_sections', []):
        ops.section("Elastic", elastic_sec['sec_tag'], elastic_sec['E'],
                   elastic_sec['A'], elastic_sec['Iz'], elastic_sec['Iy'],
                   elastic_sec['G'], elastic_sec['J'])
    
    col_count = 0
    for col in element_configs.get('force_beam_columns', []):
        ops.element("forceBeamColumn", col['tag'], col['node_i'], col['node_j'], 
                    col['transf_tag'], col['integ_tag'])
        col_count += 1
    
    beam_count = 0
    for beam in element_configs.get('elastic_beam_columns', []):
        ops.element("elasticBeamColumn", beam['tag'], beam['node_i'], beam['node_j'],
                    beam['A'], beam['E'], beam['G'], beam['J'], 
                    beam['Iy'], beam['Iz'], beam['transf_tag'])
        beam_count += 1
    
    print(f"Created {col_count} columns, {beam_count} beams")
    
    # Create shell elements
    shell_ele_count = 0
    if shell_results:
        print("\nCreating shell elements...")
        for shell_mesh in shell_results:
            config_name = shell_mesh.get('config_name', 'Unknown')
            
            shell_mat_config = None
            shell_sec_config = None
            use_zero_length = False
            zero_length_config = None
            
            for config in slab_configs:
                if config.get('name') == config_name:
                    shell_mat_config = config['shell_material_config']
                    shell_sec_config = config['shell_section_config']
                    use_zero_length = config['use_zero_length']
                    if use_zero_length:
                        zero_length_config = {
                            'material_config': config['zero_length_material_config'],
                            'directions': config['zero_length_directions'],
                            'boundary_conditions': config['zero_length_boundary_conditions'],
                            'element_start_id': config['element_start_id'],
                            'spring_node_start_id': config['spring_node_start_id']
                        }
                    break
            
            if shell_mat_config is None or shell_sec_config is None:
                continue
            
            sec_tag = shell_sec_config[1]
            
            try:
                ops.nDMaterial(shell_mat_config[0], shell_mat_config[1], 
                            shell_mat_config[2], shell_mat_config[3], shell_mat_config[4])
            except:
                pass
            
            try:
                ops.section(shell_sec_config[0], shell_sec_config[1], 
                        shell_sec_config[2], shell_sec_config[3])
            except:
                pass
            
            for elem in shell_mesh['quad4']:
                ops.element("ShellMITC4", elem['tag'], *elem['nodes'], sec_tag)
                shell_ele_count += 1
            
            for elem in shell_mesh['tri3']:
                ops.element("ASDShellT3", elem['tag'], *elem['nodes'], sec_tag)
                shell_ele_count += 1
            
            if use_zero_length and zero_length_config and zero_length_config['material_config']:
                zero_mat_tag = zero_length_config['material_config'][1]
                zero_length_material = {
                    'id': zero_mat_tag,
                    'directions': zero_length_config['directions'],
                    'config': zero_length_config['material_config']
                }
                
                node_list = [(nid, float(shell_mesh['nodes'][nid][0]), 
                                float(shell_mesh['nodes'][nid][1]), 
                                float(shell_mesh['nodes'][nid][2])) 
                                for nid in shell_mesh['nodes'].keys()]
                
                zero_result = zero_element_boundary_condition(
                    material_props=zero_length_material,
                    sections={},
                    node_list=node_list,
                    boundary_condition=zero_length_config['boundary_conditions'],
                    element_start_id=zero_length_config['element_start_id'],
                    spring_node_start_id=zero_length_config['spring_node_start_id']
                )
                
                shell_mesh['zero_length'] = zero_result
        
        print(f"Created {shell_ele_count} shell elements")
    
    # Apply loads and masses
    load_configs_to_apply = load_configs if load_configs else None
    mass_configs_to_apply = mass_configs if mass_configs else None
    
    if load_configs_to_apply or mass_configs_to_apply:
        results = apply_loads_and_masses(
            load_configs=load_configs_to_apply,
            mass_configs=mass_configs_to_apply,
            shell_meshes=shell_results,
            slab_configs=slab_configs,
            element_configs=element_configs,
            node_coords=node_coords
        )
    
    # Visualization
    if visualize:
        print("\nCreating visualization...")
        try:
            fig = opst.vis.plotly.plot_model(
                show_node_numbering=False,
                show_ele_numbering=False,
                show_ele_hover=True,
                style="surface",
                show_bc=True,
                bc_scale=0.5,
                show_outline=True
            )
            
            output_path = os.path.join(output_dir, "complete_model.html")
            fig.write_html(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Visualization error: {e}")
    
    # Summary
    all_node_tags = ops.getNodeTags()
    all_ele_tags = ops.getEleTags()
    
    print("\n" + "="*80)
    print("MODEL BUILD COMPLETE")
    print("="*80)
    print(f"Total Nodes: {len(all_node_tags)}")
    print(f"Total Elements: {len(all_ele_tags)}")
    print(f"  Columns: {col_count}")
    print(f"  Beams: {beam_count}")
    print(f"  Shell: {shell_ele_count}")
    print("="*80)
    
    # Generate complete model file
    nodal_spring_configs_to_pass = None
    for shell_mesh in shell_results:
        if 'zero_length' in shell_mesh:
            config_name = shell_mesh.get('config_name', '')
            for slab_config in slab_configs:
                if slab_config.get('name') == config_name:
                    if slab_config['use_zero_length']:
                        zero_mat_config = slab_config['zero_length_material_config']
                        if zero_mat_config:
                            nodal_spring_configs_to_pass = {
                                'material_props': {
                                    'id': zero_mat_config[1],
                                    'directions': slab_config['zero_length_directions'],
                                    'config': zero_mat_config
                                },
                                'node_list': [(nid, shell_mesh['nodes'][nid][0], 
                                            shell_mesh['nodes'][nid][1], 
                                            shell_mesh['nodes'][nid][2]) 
                                            for nid in shell_mesh['nodes'].keys()],
                                'boundary_condition': slab_config['zero_length_boundary_conditions'],
                                'element_start_id': slab_config['element_start_id'],
                                'spring_node_start_id': slab_config['spring_node_start_id']
                            }
                    break
    
    generate_complete_model_file(
        output_filepath=os.path.join(output_dir, "final_complete_model.py"),
        model_params=model_params,
        fiber_section_info=fiber_section_info,
        material_params=material_params,
        node_coords=node_coords,
        shell_meshes=shell_results,
        slab_configs=slab_configs,
        boundary_conditions=boundary_conditions,
        element_configs=element_configs,
        nodal_spring_configs=nodal_spring_configs_to_pass,
        load_configs=load_configs,
        mass_configs=mass_configs,
        diaphragm_list=diaphragm_list
    )
    
    return {
        'fiber_sections': fiber_section_info,
        'shell_meshes': shell_results,
        'total_nodes': len(all_node_tags),
        'total_elements': len(all_ele_tags)
    }


def generate_complete_model_file(output_filepath, model_params, fiber_section_info,
                                 material_params, node_coords, shell_meshes,
                                 slab_configs, boundary_conditions, element_configs,
                                 nodal_spring_configs, load_configs, mass_configs,
                                 diaphragm_list):
    """Generate final_complete_model.py with all components"""
    
    with open(output_filepath, 'w') as f:
        f.write('"""' + "\n")
        f.write("Complete OpenSeesPy Model - Auto-Generated\n")
        f.write("="*70 + "\n")
        f.write('"""' + "\n\n")
        
        f.write("import openseespy.opensees as ops\n")
        f.write("import opstool as opst\n")
        f.write("import numpy as np\n\n")
        
        # Model initialization
        f.write("# Model initialization\n")
        f.write("print('Initializing model...')\n")
        f.write("ops.wipe()\n")
        f.write(f"ops.model('basic', '-ndm', {model_params['ndm']}, '-ndf', {model_params['ndf']})\n\n")
        
        # Uniaxial materials
        f.write("# Uniaxial materials\n")
        for mat_param in material_params:
            mat_str = ", ".join([repr(p) for p in mat_param])
            f.write(f"ops.uniaxialMaterial({mat_str})\n")
        f.write(f"print('Defined {len(material_params)} materials')\n\n")
        
        # Fiber sections
        f.write("# Fiber sections\n")
        for fiber_sec in fiber_section_info:
            with open(fiber_sec['txt_path'], 'r') as sec_file:
                sec_commands = sec_file.read()
            f.write(f"\n# Section {fiber_sec['sec_tag']}\n")
            f.write(sec_commands)
            f.write("\n")
        f.write(f"print('Created {len(fiber_section_info)} fiber sections')\n\n")
        
        # Nodes
        f.write("# Frame nodes\n")
        for node_id, coords in sorted(node_coords.items()):
            x, y, z = coords
            f.write(f"ops.node({node_id}, {x}, {y}, {z})\n")
        f.write(f"print('Created {len(node_coords)} frame nodes')\n\n")
        
        if shell_meshes:
            f.write("# Shell nodes\n")
            total_shell_nodes = 0
            for shell_mesh in shell_meshes:
                config_name = shell_mesh.get('config_name', 'Unknown')
                f.write(f"\n# {config_name}\n")
                for node_id, coords in sorted(shell_mesh['nodes'].items()):
                    x, y, z = coords
                    f.write(f"ops.node({node_id}, {x}, {y}, {z})\n")
                total_shell_nodes += len(shell_mesh['nodes'])
            f.write(f"print('Created {total_shell_nodes} shell nodes')\n\n")
        
        # Boundary conditions
        f.write("# Boundary conditions\n")
        for node_id, dofs in sorted(boundary_conditions.items()):
            dof_str = ", ".join([str(d) for d in dofs])
            f.write(f"ops.fix({node_id}, {dof_str})\n")
        f.write(f"print('Applied BC to {len(boundary_conditions)} nodes')\n\n")
        
        # Rigid diaphragms
        if diaphragm_list:
            f.write("# Rigid diaphragms\n")
            for perp_dir, ret_node, *constr_nodes in diaphragm_list:
                constr_str = ", ".join([str(n) for n in constr_nodes])
                f.write(f"ops.rigidDiaphragm({perp_dir}, {ret_node}, {constr_str})\n")
            f.write(f"print('Created {len(diaphragm_list)} diaphragms')\n\n")
        
        # Transformations
        f.write("# Geometric transformations\n")
        for transf in element_configs.get('transformations', []):
            vecxz_str = ", ".join([str(v) for v in transf['vecxz']])
            f.write(f"ops.geomTransf('{transf['type']}', {transf['tag']}, {vecxz_str})\n")
        f.write("\n")
        
        # Integrations
        f.write("# Beam integrations\n")
        for integ in element_configs.get('integrations', []):
            f.write(f"ops.beamIntegration('{integ['type']}', {integ['tag']}, "
                   f"{integ['sec_tag']}, {integ['np']})\n")
        f.write("\n")
        
        # Elastic sections
        if element_configs.get('elastic_sections'):
            f.write("# Elastic sections\n")
            for elastic_sec in element_configs.get('elastic_sections', []):
                f.write(f"ops.section('Elastic', {elastic_sec['sec_tag']}, "
                       f"{elastic_sec['E']}, {elastic_sec['A']}, "
                       f"{elastic_sec['Iz']}, {elastic_sec['Iy']}, "
                       f"{elastic_sec['G']}, {elastic_sec['J']})\n")
            f.write("\n")
        
        # Beam/column elements
        f.write("# Force beam columns\n")
        for col in element_configs.get('force_beam_columns', []):
            f.write(f"ops.element('forceBeamColumn', {col['tag']}, "
                   f"{col['node_i']}, {col['node_j']}, "
                   f"{col['transf_tag']}, {col['integ_tag']})\n")
        
        col_count = len(element_configs.get('force_beam_columns', []))
        f.write(f"print('Created {col_count} columns')\n\n")
        
        f.write("# Elastic beam columns\n")
        for beam in element_configs.get('elastic_beam_columns', []):
            f.write(f"ops.element('elasticBeamColumn', {beam['tag']}, "
                   f"{beam['node_i']}, {beam['node_j']}, "
                   f"{beam['A']}, {beam['E']}, {beam['G']}, {beam['J']}, "
                   f"{beam['Iy']}, {beam['Iz']}, {beam['transf_tag']})\n")
        
        beam_count = len(element_configs.get('elastic_beam_columns', []))
        f.write(f"print('Created {beam_count} beams')\n\n")
        
        # Shell elements
        if shell_meshes:
            f.write("# Shell elements\n")
            total_shell_elements = 0
            
            for shell_mesh in shell_meshes:
                config_name = shell_mesh.get('config_name', 'Unknown')
                
                shell_mat_config = None
                shell_sec_config = None
                
                for config in slab_configs:
                    if config.get('name') == config_name:
                        shell_mat_config = config['shell_material_config']
                        shell_sec_config = config['shell_section_config']
                        break
                
                if shell_mat_config is None or shell_sec_config is None:
                    continue
                
                f.write(f"\n# {config_name}\n")
                
                mat_str = ", ".join([repr(p) for p in shell_mat_config])
                f.write(f"ops.nDMaterial({mat_str})\n")
                
                sec_str = ", ".join([repr(p) for p in shell_sec_config])
                f.write(f"ops.section({sec_str})\n\n")
                
                sec_tag = shell_sec_config[1]
                
                for elem in shell_mesh['quad4']:
                    node_str = ", ".join([str(n) for n in elem['nodes']])
                    f.write(f"ops.element('ShellMITC4', {elem['tag']}, {node_str}, {sec_tag})\n")
                
                if shell_mesh['tri3']:
                    for elem in shell_mesh['tri3']:
                        node_str = ", ".join([str(n) for n in elem['nodes']])
                        f.write(f"ops.element('ASDShellT3', {elem['tag']}, {node_str}, {sec_tag})\n")
                
                elem_count = len(shell_mesh['quad4']) + len(shell_mesh['tri3'])
                total_shell_elements += elem_count
            
            f.write(f"\nprint('Created {total_shell_elements} shell elements')\n\n")
        
        # Zero-length springs
        if nodal_spring_configs:
            f.write("# Zero-length springs\n")
            
            mat_config = nodal_spring_configs['material_props']['config']
            mat_str = ", ".join([repr(p) for p in mat_config])
            f.write(f"ops.uniaxialMaterial({mat_str})\n\n")
            
            mat_id = nodal_spring_configs['material_props']['id']
            directions = nodal_spring_configs['material_props']['directions']
            boundary_condition = nodal_spring_configs['boundary_condition']
            element_start_id = nodal_spring_configs['element_start_id']
            spring_node_start_id = nodal_spring_configs['spring_node_start_id']
            
            spring_count = 0
            for i, (node_id, x, y, z) in enumerate(nodal_spring_configs['node_list']):
                spring_node_id = spring_node_start_id + i
                elem_id = element_start_id + i
                
                bc_str = ", ".join([str(d) for d in boundary_condition])
                dir_str = ", ".join([str(d) for d in directions])
                
                f.write(f"ops.node({spring_node_id}, {x}, {y}, {z})\n")
                f.write(f"ops.fix({spring_node_id}, {bc_str})\n")
                f.write(f"ops.element('zeroLength', {elem_id}, {node_id}, {spring_node_id}, "
                       f"'-mat', {mat_id}, '-dir', {dir_str})\n")
                
                spring_count += 1
            
            f.write(f"\nprint('Created {spring_count} springs')\n\n")
        
        # Loads
        if load_configs:
            f.write("# Loads\n")
            
            if 'time_series' in load_configs:
                for ts in load_configs['time_series']:
                    if ts['type'] == 'Linear':
                        f.write(f"ops.timeSeries('Linear', {ts['tag']})\n")
                    elif ts['type'] == 'Constant':
                        f.write(f"ops.timeSeries('Constant', {ts['tag']})\n")
                f.write("\n")
            
            if 'patterns' in load_configs:
                for pattern in load_configs['patterns']:
                    f.write(f"ops.pattern('Plain', {pattern['tag']}, {pattern['ts_tag']})\n")
                f.write("\n")
            
            if 'nodal_loads' in load_configs:
                for load_group in load_configs['nodal_loads']:
                    for load in load_group['loads']:
                        force_str = ", ".join([str(f) for f in load['forces']])
                        f.write(f"ops.load({load['node']}, {force_str})\n")
                f.write("\n")
            
            if 'beam_uniform_loads' in load_configs:
                for load_group in load_configs['beam_uniform_loads']:
                    for load in load_group['loads']:
                        elem_str = str(load['elements'])
                        f.write(f"opst.pre.transform_beam_uniform_load({elem_str}, "
                               f"wy={load['wy']}, wz={load['wz']})\n")
                f.write("\n")
            
            if 'beam_point_loads' in load_configs:
                for load_group in load_configs['beam_point_loads']:
                    for load in load_group['loads']:
                        f.write(f"opst.pre.transform_beam_point_load([{load['element']}], "
                               f"py={load['py']}, pz={load['pz']}, xl={load['xl']})\n")
                f.write("\n")
            
            if 'shell_surface_loads' in load_configs:
                for load_group in load_configs['shell_surface_loads']:
                    for load in load_group['loads']:
                        for shell_mesh in shell_meshes:
                            if shell_mesh.get('config_name') == load['mesh_name']:
                                if load['elements'] is None:
                                    element_tags = [elem['tag'] for elem in shell_mesh['quad4']]
                                    element_tags += [elem['tag'] for elem in shell_mesh['tri3']]
                                else:
                                    element_tags = load['elements']
                                
                                f.write(f"opst.pre.transform_surface_uniform_load("
                                       f"ele_tags={element_tags}, p={load['pressure']})\n")
                                break
                f.write("\n")
            
            f.write("print('Loads applied')\n\n")
        
        # Masses
        if mass_configs:
            f.write("# Masses\n")
            f.write("nodal_masses = {}\n\n")
            
            if 'nodal_mass' in mass_configs:
                for item in mass_configs['nodal_mass']:
                    node_id = item['node']
                    mass_value = item['mass']
                    f.write(f"if {node_id} not in nodal_masses:\n")
                    f.write(f"    nodal_masses[{node_id}] = 0.0\n")
                    f.write(f"nodal_masses[{node_id}] += {mass_value}\n")
                f.write("\n")
            
            f.write("for node_id, mass_value in nodal_masses.items():\n")
            f.write("    if mass_value > 0:\n")
            f.write("        ops.mass(node_id, mass_value, mass_value, mass_value, 0.0, 0.0, 0.0)\n\n")
            
            f.write("print('Masses applied')\n\n")
        
        # Footer
        f.write("print('\\n' + '='*70)\n")
        f.write("print('MODEL COMPLETE')\n")
        f.write("print('='*70)\n")
        f.write("print(f'Nodes: {len(ops.getNodeTags())}')\n")
        f.write("print(f'Elements: {len(ops.getEleTags())}')\n")
        f.write("print('='*70)\n")
    
    print(f"\nGenerated: {output_filepath}")


