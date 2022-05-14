import bpy
import os
import yaml

for area in bpy.context.screen.areas:
    if area.type == 'CLIP_EDITOR':
        break
else:
    print('No area found')
    exit(1)

override = bpy.context.copy()
override['area'] = area

data_path = os.path.join(bpy.path.abspath('//'), '..', 'tracks.yaml')

with open(data_path, 'r') as fp:
    tracks = yaml.safe_load(fp)

clip = bpy.data.movieclips[0]
width = clip.size[0]  # 848
height = clip.size[1]  # 480

for track in tracks:
    points = track['points']

    def p2l(p):
        return p['u']/width, p['v']/height

    bpy.ops.clip.add_marker(override, location=p2l(points[0]))
    bpy.data.movieclips[0].tracking.tracks.active.name="id_" + str(track['id'])
    marker = clip.tracking.objects[0].tracks[-1].markers
    for point in points:
        marker.insert_frame(point['t'], co=p2l(point))
