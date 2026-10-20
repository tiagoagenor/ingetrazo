// Showcase: a small pavilion, built through IngeTrazo's scripting API.
//
// Port of scripts/create_architectural_showcase.py (the Python console demo).
// Run it from the script console; everything it creates arrives as ONE undo
// step. Differences from the Python original:
//   * every element becomes a named group (the Python one left the slab and
//     the walls as loose faces and only the columns as groups);
//   * layers are created by assign_layer;
//   * the IFC tags (bim.tag_faces / bim.tag_group) are not ported: there is no
//     tagging command in the Rust model API yet.

const CONCRETE = [0.62, 0.60, 0.58];
const GLASS = [0.55, 0.75, 0.85];
const TIMBER = [0.55, 0.38, 0.22];

const W = 8.0;   // pavilion footprint and height, metres
const D = 6.0;
const H = 3.0;

// Six quads of an axis-aligned box; returns the new faces.
fn box_faces(x0, y0, z0, x1, y1, z1, color) {
    let quads = [
        // Floor wound clockwise seen from above: its outward normal points
        // DOWN, so the signed volume of the solid is not counted backwards.
        [vec3(x0, y1, z0), vec3(x1, y1, z0), vec3(x1, y0, z0), vec3(x0, y0, z0)],
        [vec3(x0, y0, z1), vec3(x1, y0, z1), vec3(x1, y1, z1), vec3(x0, y1, z1)],
        [vec3(x0, y0, z0), vec3(x1, y0, z0), vec3(x1, y0, z1), vec3(x0, y0, z1)],
        [vec3(x1, y0, z0), vec3(x1, y1, z0), vec3(x1, y1, z1), vec3(x1, y0, z1)],
        [vec3(x1, y1, z0), vec3(x0, y1, z0), vec3(x0, y1, z1), vec3(x1, y1, z1)],
        [vec3(x0, y1, z0), vec3(x0, y0, z0), vec3(x0, y0, z1), vec3(x0, y1, z1)],
    ];
    let faces = [];
    for q in quads {
        faces.push(add_face(q, #{ color: color }));
    }
    faces
}

// A box as a named group on a layer.
fn element(name, layer, x0, y0, z0, x1, y1, z1, color) {
    let faces = box_faces(x0, y0, z0, x1, y1, z1, color);
    assign_layer(faces, layer);
    make_group(faces, name)
}

// Floor slab.
element("Losa de piso", "Estructura", 0.0, 0.0, -0.20, W, D, 0.0, CONCRETE);

// Four corner columns.
let corners = [[0.15, 0.15], [W - 0.45, 0.15], [0.15, D - 0.45], [W - 0.45, D - 0.45]];
let n = 1;
for c in corners {
    element(`Columna ${n}`, "Estructura", c[0], c[1], 0.0, c[0] + 0.30, c[1] + 0.30, H, CONCRETE);
    n += 1;
}

// Two solid walls and a glazed front (loose: a single translucent face).
element("Muro posterior", "Muros", 0.0, D - 0.15, 0.0, W, D, H, CONCRETE);
element("Muro lateral", "Muros", 0.0, 0.0, 0.0, 0.15, D, H, CONCRETE);
let front = add_face([vec3(0.6, 0.02, 0.0), vec3(W, 0.02, 0.0), vec3(W, 0.02, H), vec3(0.6, 0.02, H)], #{ color: GLASS });
set_opacity(front, 0.35);
assign_layer(front, "Muros");

// Timber roof slab.
element("Cubierta", "Estructura", -0.30, -0.30, H, W + 0.30, D + 0.30, H + 0.15, TIMBER);

print(`Pabellon listo: ${face_count()} caras sueltas, ${groups().len()} grupos.`);
for g in groups() {
    print(`  ${g.name}: ${g.faces().len()} caras, capa ${g.layer}`);
}
print("Ctrl+Z deshace todo el pabellon de una vez.");
