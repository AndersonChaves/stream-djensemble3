import numpy as np

class QuadTree:
    def __init__(self, data, min_purity, cur_depth=0, max_depth=4):
        self.min_purity = min_purity
        self.max_depth  = max_depth
        self.cur_depth  = cur_depth
        self.data = data

        if self.is_partitionable():
            self.end_point = len(data)-1, len(data[0])-1
            self.mid_point = (len(data)-1) // 2, (len(data[0])-1) // 2
            self.quad = self.partitionate_tree()
            self.is_leaf = False
        else:
            self.quad = None
            self.is_leaf = True

    def is_partitionable(self):
        if (self.calculate_purity() < self.min_purity) and \
           (self.cur_depth+1 <= self.max_depth) and \
           (len(self.data) >= 2) and \
           (len(self.data[0]) >= 2):
            return True
        else:
            return False

    def partitionate_tree(self):
        data = self.data
        end_x, end_y = self.end_point
        mid_x, mid_y = self.mid_point

        quad = {}
        quad["1"] = QuadTree(data[0:mid_x+1, 0:mid_y+1], self.min_purity,
                               self.cur_depth+1, self.max_depth)
        quad["2"] = QuadTree(data[0:mid_x + 1, mid_y + 1:], self.min_purity,
                             self.cur_depth + 1, self.max_depth)
        quad["3"] = QuadTree(data[mid_x + 1:, 0:mid_y + 1], self.min_purity,
                             self.cur_depth + 1, self.max_depth)
        quad["4"] = QuadTree(data[mid_x+1:, mid_y+1:], self.min_purity,
                               self.cur_depth+1, self.max_depth)
        return quad

    def calculate_purity(self):
        if len(self.data) == 0 or len(self.data[0]):
            return 1
        print("Calculating puriity: ", self.data.shape)
        print("Calculating puriity: ", self.data)
        b_count = np.bincount(self.data.flatten())
        highest_frequency = b_count[b_count.argmax()]
        purity = highest_frequency / self.data.size
        return purity

    def get_quadrant_at_position(self, x: int, y: int) -> np.array:
        '''
        :param x: Desired Row position
        :param y: Desired Column position
        :return: The quadrant that intersects with the specified position.
        '''
        if self.is_leaf:
            return self.data
        elif x <= self.mid_point[0] and y <= self.mid_point[1]:
            return self.quad["1"].get_quadrant_at_position(x, y)
        elif x > self.mid_point[0] and y <= self.mid_point[1]:
            return self.quad["3"].get_quadrant_at_position(self.mid_point[0] + x, y)
        elif x <= self.mid_point[0] and y > self.mid_point[1]:
            return self.quad["2"].get_quadrant_at_position(x, self.mid_point[1] + y)
        else:
            return self.quad["4"].get_quadrant_at_position(self.mid_point[0] + x,
                                                              self.mid_point[1] + y)

    def get_all_quadrants(self) -> np.array:
        if self.is_leaf:
            partitions = np.zeros(shape=self.data.shape)
            return partitions
        else:
            part1 = self.quad["1"].get_all_quadrants()
            part2 = self.quad["2"].get_all_quadrants() + part1.max() + 1
            part3 = self.quad["3"].get_all_quadrants() + part2.max() + 1
            part4 = self.quad["4"].get_all_quadrants() + part3.max() + 1

            full = np.concatenate(
                (
                    np.concatenate((part1, part2), axis=1),
                    np.concatenate((part3, part4), axis=1)
                ),
                axis=0
            )
            return full

    def get_all_quadrant_limits(self):
        full_partitions = self.get_all_quadrants()
        quad_dict = {}
        for i in range(int(np.max(full_partitions)+1)):
            indexes = np.argwhere(full_partitions == i)
            quad_dict[str(i)] = {}
            quad_dict[str(i)]["start"] = indexes[0]
            quad_dict[str(i)]["end"] = indexes[-1]
        return quad_dict

def test_quad_tree():
    data = [[1, 1, 2, 2],
            [1, 1, 2, 2],
            [2, 3, 2, 1],
            [3, 1, 2, 1]]
    data = np.array(data)
    quadtree = QuadTree(data, min_purity=1)
    assert (
        np.array_equal(quadtree.get_quadrant_at_position(0, 2), np.array([[2, 2], [2, 2]]))
    )
    assert (
        np.array_equal(quadtree.get_quadrant_at_position(2, 0), np.array([[3]]))
    )
    assert (
        np.array_equal(quadtree.get_quadrant_at_position(3, 3), np.array([[1]])) and
        not np.array_equal(quadtree.get_quadrant_at_position(3, 3), np.array([[2, 1], [2, 1]]))
    )

    assert (
        np.array_equal(quadtree.get_all_quadrants(), np.array([[0, 0, 1, 1],
                                                               [0, 0, 1, 1],
                                                               [2, 3, 6, 7],
                                                               [4, 5, 8, 9]]))
    )

    assert(
        np.array_equal(quadtree.get_all_quadrant_limits()["5"]["start"], np.array([3, 1])) and
        np.array_equal(quadtree.get_all_quadrant_limits()["5"]["end"], np.array([3, 1]))
    )

    quadtree = QuadTree(data, min_purity=0.5)
    assert (
            not np.array_equal(quadtree.get_quadrant_at_position(3, 3), np.array([[1]])) and
            np.array_equal(quadtree.get_quadrant_at_position(3, 3), np.array([[2, 1], [2, 1]]))
    )

    assert (
        np.array_equal(quadtree.get_all_quadrants(), np.array([[0, 0, 1, 1],
                                                               [0, 0, 1, 1],
                                                               [2, 2, 3, 3],
                                                               [2, 2, 3, 3]]))
    )



    print("All tests passed")

if __name__ == "__main__":
    test_quad_tree()
