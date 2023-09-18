import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from itertools import cycle
from .quad_tree import QuadTree
from core.clustering import IClustering
from .tile import Tile

class Tiling():
    def __init__(self, config, query_start):
        self.config = config        

    def run(self, clustering: IClustering, target_dataset):
                
        if self.config["strategy"] == "yolo":
            self.tiling_frame, self.tile_dict = create_yolo_tiling(
                clustering.clustering_matrix, self.config["min_purity_rate"]
            )
        elif self.config["strategy"] == "quadtree":
            self.tiling_frame, self.tile_dict = create_quadtree_tiling(
                clustering.clustering_matrix, self.config["min_purity_rate"]
            )

        tiles = []
        for tile_id, value in self.tile_dict.items():            
            start, end = value['start'], value['end']            

            if clustering.embedding_matrix is not None:
                centroid_coord = calculate_centroid(clustering.embedding_matrix, start, end)                                
                centroid_series = target_dataset[:, centroid_coord[0], centroid_coord[1]] 
            tiles.append(Tile(tile_id, (start, end), 
                centroid_coord, centroid_series, offset=self.query_start))
        self.tiles = tiles        

    def get_number_of_tiles(self):
        return len(self.tiles)

def expand_tile(tiling: np.array, clustering: np.array, start: tuple,
                tile_id, max_impurity_rate = 0.05):
    impurity = 0

    lat, long = clustering.shape
    tile_cluster = clustering[start]
    cursor = end = start
    tiling[cursor] = tile_id

    dir_remaining = 4
    directions = cycle([(1, 0), (0, 1), (-1, 0), (0, -1)])
    skip_list = []
    max_impurity = 1
    for dir_step in directions:
        if len(skip_list) == 4:
            break
        if dir_step in skip_list:
            continue
        updates = True
        current_impurity = impurity
        #---EXPAND RIGHT---------------------------
        if dir_step == (1, 0) and end[1]+1 < long: #todo: Create Tile Objetct for Expansion
            end = end[0], end[1]+1
            x_start, y_start = start[0], end[1]
            x_end, y_end = end
            for i in range(x_start, x_end+1):
                cursor = i, y_start
                if tiling[cursor] != -1:
                    end = end[0], end[1] - 1
                    updates = False
                    break
                elif clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        end = end[0], end[1] - 1 # Undo Expantion
                        updates=False
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(x_start, x_end + 1):
                    cursor = i, y_start
                    tiling[cursor] = tile_id
        # ---EXPAND DOWN---------------------------
        elif dir_step == (0, 1) and end[0]+1 < lat:
            end = end[0]+1, end[1]
            x_start, y_start = end[0], start[1]
            x_end, y_end = end
            for i in range(y_start, y_end + 1):
                cursor = x_start, i
                if tiling[cursor] != -1:
                    updates = False
                    end = end[0] - 1, end[1]
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        updates=False
                        end = end[0] - 1, end[1]
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(y_start, y_end + 1):
                    cursor = x_start, i
                    tiling[cursor] = tile_id
        # ---EXPAND LEFT---------------------------
        elif dir_step == (-1, 0) and start[1]-1 >= 0:
            start = start[0], start[1]-1
            x_start, y_start = start
            x_end, y_end = end[0], start[1]+1
            for i in range(x_start, x_end + 1):
                cursor = i, y_start
                if tiling[cursor] != -1:
                    start = start[0], start[1] + 1
                    updates = False
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        start = start[0], start[1]+1
                        updates = False
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(x_start, x_end + 1):
                    cursor = i, y_start
                    tiling[cursor] = tile_id

        # ---EXPAND UP---------------------------
        elif dir_step == (0, -1) and start[0]-1 >= 0:
            start = start[0]-1, start[1]
            x_start, y_start = start
            x_end, y_end = start[0], end[1]
            for i in range(y_start, y_end + 1):
                cursor = x_start, i
                if tiling[cursor] != -1:
                    updates = False
                    start = start[0] + 1, start[1]
                    break
                if clustering[cursor] != tile_cluster:
                    if current_impurity >= max_impurity:
                        updates=False
                        start = start[0] + 1, start[1]
                        break
                    else:
                        current_impurity += 1
            if updates:
                for i in range(y_start, y_end + 1):
                    cursor = x_start, i
                    tiling[cursor] = tile_id
        #----------------------------------------
        else:
            updates = False
        if updates:
            impurity = current_impurity
            tile_size = (abs(start[0] - end[0])+1) * (abs(start[1] - end[1])+1)
            max_impurity = round((tile_size*max_impurity_rate), 0)
            skip_list = []
        else:
            skip_list.append(dir_step)
            dir_remaining -= 1
    return {"start": start, "end": end}

def create_yolo_tiling(clustering: np.array, min_purity_rate: int):
    shp = clustering.shape
    lat, long = shp
    tiling = np.full(shp, -1)
    x = 0
    tile_id = 1
    tile_dict = {}
    while x < lat:
        y = 0
        while y < long:
            if tiling[x, y] == -1:
                tile_dict[tile_id] = expand_tile(tiling, clustering, (x, y), tile_id,
                                                 max_impurity_rate=1 - min_purity_rate)
                tile_id += 1
            y += 1
        x += 1
    return tiling, tile_dict

def create_quadtree_tiling(clustering: np.array, min_purity_rate):
    quadtree = QuadTree(clustering, min_purity=min_purity_rate)
    tiling = quadtree.get_all_quadrants()
    tiling_dict = quadtree.get_all_quadrant_limits()
    return tiling, tiling_dict

def calculate_centroid(clustering, start, end):
    tile_embedd = clustering[start[0]:end[0]+1, start[1]:end[1]+1]
    if len(tile_embedd.shape) == 3:
        t_shp = tile_embedd.shape
    else:
        t_shp = tile_embedd.shape + tuple([1])

    centroid_gld = [np.average(tile_embedd[:, :, p]) for p in range(t_shp[2])]
    
    c, _ = pairwise_distances_argmin_min(np.reshape(centroid_gld, (1, -1)),
                                     np.reshape(tile_embedd, (t_shp[0] * t_shp[1], t_shp[2])))
    #print("Centroid: ", c)
    centroid = c // t_shp[1], c % t_shp[1]
    # Returns the centroid position relative to the tile
    return centroid