from palette_extraction.structs import Ray, Triangle, Mesh

path = "../../palette-refine/obj_original/02-mesh-obj-files.obj"

m = Mesh()
m.load_from_file(path)
t = m.construct_triangles()
print(m.vertices)
print(m.faces)
m.save_to_file("./spam.txt")
