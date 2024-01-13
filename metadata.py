from series_pattern import SeriesPattern

series_patterns = {
    1: SeriesPattern("linear"),
    2: SeriesPattern("sinusoidal"),
    3: SeriesPattern("random_walk"),
    4: SeriesPattern("exponential_growth")
}

patterns_frame = [
    [1, 2, 2, 3],
    [1, 2, 2, 3],
    [4, 4, 4, 4],
    [4, 4, 4, 4]
]

block_length = 5
time_dimension_len = 10000