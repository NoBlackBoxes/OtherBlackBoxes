# Import python libraries
import os
import numpy as np



# Import blender python libraries
import bpy



# Get books collection
books = bpy.data.collections["Books"]

# Diable active selection
bpy.context.active_object.select_set(False)


 # Get the prototype book and make active
book = books.objects["Book"]
for child in book.children:
    bpy.context.view_layer.objects.active = child
    bpy.context.active_object.select_set(True)
bpy.context.view_layer.objects.active = book
bpy.context.active_object.select_set(True)


nbook = 0
for i in range(5):
    nbook = nbook + 1

    # Make a copy (duplicate) and rename
    bpy.ops.object.duplicate(linked=False)
    new_book = bpy.context.active_object
    new_book.name = "New_Book" + str(nbook)

    # Get location
    location = new_book.location
    new_book.location.z = location.z + 0.4
    new_book.scale.x = np.random.rand()
    new_book.scale.y = np.random.rand()




# Repeat!
num_books = 10

# Create new cover materials (copy of leather)
# new_material = bpy.data.materials.new("new"")
# new_material['shader'] = vk_material.shader