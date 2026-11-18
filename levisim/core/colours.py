# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://stackoverflow.com/questions/64369710/what-are-the-hex-codes-of-matplotlib-tab10-palette#64369873
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
light_purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
grey = '#7f7f7f'
dark_yellow = '#bcbd22'
light_blue = '#17becf'
purple = '#8031a7'

matplotlib_colours = {
    'blue': blue,
    'orange': orange,
    'green': green,
    'red': red,
    'light_purple': light_purple,
    'brown': brown,
    'pink': pink,
    'grey': grey,
    'dark_yellow': dark_yellow,
    'light_blue': light_blue,
    'purple': purple,
}

# maxed out RGB colours
RGB_red = '#ff0000'
RGB_green = '#00ff00'
RGB_blue = '#0000ff'
RGB_cyan = '#00ffff'
RGB_magenta = '#ff00ff'
RGB_yellow = '#ffff00'

RGB_colours = {
    'RGB_red': RGB_red,
    'RGB_green': RGB_green,
    'RGB_blue': RGB_blue,
    'RGB_cyan': RGB_cyan,
    'RGB_magenta': RGB_magenta,
    'RGB_yellow': RGB_yellow,
}

# the pipe | concatenates the two dicts
all_colours_dict = matplotlib_colours | RGB_colours
all_colours_list = [
    # matplotlib_colours
    blue, orange, green,
    red, light_purple, brown,
    pink, grey, dark_yellow,
    light_blue, purple,
    # RGB colours
    RGB_red, RGB_green, RGB_blue,
    RGB_cyan, RGB_magenta, RGB_yellow,
]

# visually distinct colour generator!
# https://mokole.com/palette.html
darkslategray_16 = '#2f4f4f'
maroon2_16 = '#7f0000'
midnightblue_16 = '#191970'
green_16 = '#008000'
red_16 = '#ff0000'
orange_16 = '#ffa500'
yellow_16 = '#ffff00'
mediumblue_16 = '#0000cd'
lime_16 = '#00ff00'
mediumspringgreen_16 = '#00fa9a'
darksalmon_16 = '#e9967a'
aqua_16 = '#00ffff'
fuchsia_16 = '#ff00ff'
dodgerblue_16 = '#1e90ff'
plum_16 = '#dda0dd'
deeppink_16 = '#ff1493'

distinct_16_colours_list = [
    darkslategray_16,
    maroon2_16,
    midnightblue_16,
    green_16,
    red_16,
    orange_16,
    yellow_16,
    mediumblue_16,
    lime_16,
    mediumspringgreen_16,
    darksalmon_16,
    aqua_16,
    fuchsia_16,
    dodgerblue_16,
    plum_16,
    deeppink_16,
]

# these are bright and vary decently
colours_50 = [
    "#FF5733", "#33FF57", "#3357FF", "#FF33A6", "#33FFF6",
    "#F6FF33", "#FF9D33", "#33FF9D", "#9D33FF", "#FF3333",
    "#33FF33", "#3333FF", "#FF5733", "#57FF33", "#5733FF",
    "#A6FF33", "#33A6FF", "#FF33F6", "#F633FF", "#33F6FF",
    "#9DFF33", "#FF339D", "#FF3339", "#39FF33", "#3339FF",
    "#33FFF9", "#FF33C1", "#C1FF33", "#33C1FF", "#FF9D57",
    "#57FF9D", "#9D57FF", "#FF579D", "#579DFF", "#33FF63",
    "#FF6333", "#6333FF", "#33FFA6", "#FFA633", "#A633FF",
    "#FF33FF", "#33FFFF", "#FFFF33", "#FF8C00", "#8C00FF",
    "#FF00A0", "#00A0FF", "#8CFF00", "#00FF8C", "#FF5732"
]

# keep black and white separate away from plotting use
black = '#000000'
white = '#ffffff'

# https://www.color-hex.com/color-palette/1054396
blues = [
    '#000066',
    '#000099',
    '#0033cc',
    '#0099ff',
    '#66ffff',
]
# https://www.color-hex.com/color-palette/1047421
dreamlands = [
    '#a1ffb1',
    '#a1ffd3',
    '#a1e7ff',
    '#a1cdff',
    '#a1a7ff',
]

# https://waldyrious.net/viridis-palette-generator/
discrete_viridis = {
    2: ['#fde725', '#440154'],
    3: ['#fde725', '#21918c', '#440154'],
    4: ['#fde725', '#35b779', '#31688e', '#440154'],
    5: ['#fde725', '#5ec962', '#21918c', '#3b528b', '#440154'],
    6: ['#fde725', '#7ad151', '#22a884', '#2a788e', '#414487', '#440154'],
    7: ['#fde725', '#90d743', '#35b779', '#21918c', '#31688e', '#443983', '#440154'],
    8: ['#fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d', '#46327e', '#440154'],
    9: ['#fde725', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c728e', '#3b528b', '#472d7b', '#440154'],
   10: ['#fde725', '#b5de2b', '#6ece58', '#35b779', '#1f9e89', '#26828e', '#31688e', '#3e4989', '#482878', '#440154'],
}
discrete_plasma = {
    2: ['#f0f921', '#0d0887'],
    3: ['#f0f921', '#cc4778', '#0d0887'],
    4: ['#f0f921', '#ed7953', '#9c179e', '#0d0887'],
    5: ['#f0f921', '#f89540', '#cc4778', '#7e03a8', '#0d0887'],
    6: ['#f0f921', '#fca636', '#e16462', '#b12a90', '#6a00a8', '#0d0887'],
    7: ['#f0f921', '#fdb42f', '#ed7953', '#cc4778', '#9c179e', '#5c01a6', '#0d0887'],
    8: ['#f0f921', '#febd2a', '#f48849', '#db5c68', '#b83289', '#8b0aa5', '#5302a3', '#0d0887'],
    9: ['#f0f921', '#fdc527', '#f89540', '#e66c5c', '#cc4778', '#aa2395', '#7e03a8', '#4c02a1', '#0d0887'],
   10: ['#f0f921', '#fdca26', '#fb9f3a', '#ed7953', '#d8576b', '#bd3786', '#9c179e', '#7201a8', '#46039f', '#0d0887'],
}
discrete_magma = {
    2: ['#fcfdbf', '#000004'],
    3: ['#fcfdbf', '#b73779', '#000004'],
    4: ['#fcfdbf', '#f1605d', '#721f81', '#000004'],
    5: ['#fcfdbf', '#fc8961', '#b73779', '#51127c', '#000004'],
    6: ['#fcfdbf', '#fe9f6d', '#de4968', '#8c2981', '#3b0f70', '#000004'],
    7: ['#fcfdbf', '#feb078', '#f1605d', '#b73779', '#721f81', '#2c115f', '#000004'],
    8: ['#fcfdbf', '#febb81', '#f8765c', '#d3436e', '#982d80', '#5f187f', '#221150', '#000004'],
    9: ['#fcfdbf', '#fec488', '#fc8961', '#e75263', '#b73779', '#832681', '#51127c', '#1d1147', '#000004'],
   10: ['#fcfdbf', '#feca8d', '#fd9668', '#f1605d', '#cd4071', '#9e2f7f', '#721f81', '#440f76', '#180f3d', '#000004'],
}


# ------------------------------------------------------------------
# Public names exported by this module.
# Colour names and palettes used by the plotting helpers.
# ------------------------------------------------------------------
__all__ = [
    'RGB_blue',
    'RGB_colours',
    'RGB_cyan',
    'RGB_green',
    'RGB_magenta',
    'RGB_red',
    'RGB_yellow',
    'all_colours_dict',
    'all_colours_list',
    'aqua_16',
    'black',
    'blue',
    'blues',
    'brown',
    'colours_50',
    'dark_yellow',
    'darksalmon_16',
    'darkslategray_16',
    'deeppink_16',
    'discrete_magma',
    'discrete_plasma',
    'discrete_viridis',
    'distinct_16_colours_list',
    'dodgerblue_16',
    'dreamlands',
    'fuchsia_16',
    'green',
    'green_16',
    'grey',
    'light_blue',
    'light_purple',
    'lime_16',
    'maroon2_16',
    'matplotlib_colours',
    'mediumblue_16',
    'mediumspringgreen_16',
    'midnightblue_16',
    'orange',
    'orange_16',
    'pink',
    'plum_16',
    'purple',
    'red',
    'red_16',
    'white',
    'yellow_16',
]
