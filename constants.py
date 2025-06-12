import numpy as np

names = {0: "Do nothing", 1: "Get water",
         2: "Get food", 3: "Get sodium", 4: "Get warmth", }

all_colors = {'blue':    '#377eb8',
              'orange':  '#ff7f00',
              'green':   '#4daf4a',
              'pink':    '#f781bf',
              'brown':   '#a65628',
              'purple':  '#984ea3',
              'gray':    '#999999',
              'red':     '#e41a1c',
              'yellow':  '#dede00',
              'black': '#000000'
              }

colors = {index: all_colors[key]
          for index, key in enumerate(list(all_colors.keys()))}


def konidaris_function_sym(sigma=1, rho=1):
    if 0 <= sigma and sigma < 1:
        return 1-sigma**(np.tan(rho*np.pi/2))
    elif sigma >= 1 and sigma <= 2:
        return -1 + (2-sigma)**(np.tan(rho*np.pi/2))
    else:
        return -1


nb_to_plot = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

functions_konidaris = {}
for i, value_i in enumerate(nb_to_plot):
    functions_konidaris[i] = lambda x, val=value_i: konidaris_function_sym(
        x, val)

names_konidaris = {i: el for i, el in enumerate(nb_to_plot)}
