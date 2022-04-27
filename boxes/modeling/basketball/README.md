# Modeling : Basketball

Create a realistic basketball model for use in a game

## Steps

- Add a UV spehere and change to 16 segments (check is color texture looks OK?)

- Delete all vertices except for one octent (1/8th) of spehere
- Add two cuts using "knife" tool to make basketball seam pattern
- Select seam edges
- Add a mirror modifier to restore full sphere (X, Y, and X axis)
  - Go to object mode and apply modifier, then return to Edit mode
- Use "bevel" tool to create seam width (chenage segmeants to 7)
- Deselect the outermoust segment of seam (Select-more/less-less)
- Use "resize" tool to create seam depth (same along all axes)
- Return to object mode and add a subdivision modifier to increase poly count, 3?
- Add a new "ball" material (orange?)
- Add a new "seams" material for the seams (black?)
- Assign the "seams" material to seams, go to edit mode, with seams selected, click assign
- Add some lights!
- Add a skin texture: Shader editor, ball materials, add Voroni texture (for height of bump vector node) and send to shader normals