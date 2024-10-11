from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.floorplan import Floorplan

c = Controller()


floorplan = Floorplan()

floorplan.init_scene(scene='4b', layout=1)

c.add_ons.extend([floorplan])
# Create the scene.
c.communicate([])
c.communicate([])

occupancy_map = OccupancyMap()
c.add_ons.append(occupancy_map)
occupancy_map.generate(cell_size=0.5, once=False)
occ = occupancy_map


c.communicate([])


for x_index in range(occ.occupancy_map.shape[0]):
    for z_index in range(occ.occupancy_map.shape[1]):
        x = float(occ.positions[x_index, z_index][0])
        z = float(occ.positions[x_index, z_index][1])
        print(x, z)


print(occupancy_map.occupancy_map)


c.communicate({"$type": "terminate"})