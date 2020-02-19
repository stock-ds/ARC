"""
The Abstraction and Reasoning Corpus (ARC) data set predictor.

General pipeline and future ideas:
load data
split up problems
categorize by dimension:
    x*x in -> x*x out
    linear dimension predictor
    custom dimension out based on image
find most correlated pixels
create best features

Completed prediction attempt:
simple input - output model using Trees (where  x*x in -> x*x out)
"""
import json
import pandas as pd
from pandas.io.json import json_normalize
import os
import numpy as np
from sklearn import tree, svm, linear_model
from collections import Counter
import lightgbm
import matplotlib.pyplot as plt


def load_data(base_folder="ARC/data/training"):
    df = pd.DataFrame()
    for n, file in enumerate(os.listdir(base_folder)):
        with open(base_folder+"/"+file, "r") as j:
            loadedset = json.loads(j.read())

        traintemp = json_normalize(loadedset, "train")
        traintemp['id'] = n
        traintemp['train'] = True
        traintemp['file'] = file

        testtemp = json_normalize(loadedset, "test")
        testtemp['id'] = n
        testtemp['train'] = False
        testtemp['file'] = file

        df = df.append(traintemp.append(testtemp))

    df = df.reset_index(drop=True)
    df['Prediction'] = None
    df['Type'] = None
    df_ids = set(df['id'])

    # identify output dimensions and categorize them

    df['In dim'] = df['input'].apply(lambda x: np.array([len(x), len(x[0])]))
    df['Out dim'] = df['output'].apply(lambda x: np.array([len(x), len(x[0])]))

    size_model = linear_model.LinearRegression()
    for id_ in df_ids:
        # print(id_)
        id_train = (df['id'] == id_) & (df['train'] == True)

        if np.array_equal(list(df.loc[id_train, 'In dim']), list(df.loc[id_train, 'Out dim'])):
            df.loc[(df['id'] == id_), 'Prediction dim'] = df.loc[(df['id'] == id_), 'In dim']
            df.loc[(df['id'] == id_), 'Prediction'] = \
                df.loc[(df['id'] == id_), 'Prediction dim'].apply(lambda x: np.zeros(x, np.int8))
            df.loc[(df['id'] == id_), 'Type'] = "Same_out"
        else:
            X = pd.DataFrame(list(df.loc[(df['id'] == id_) & (df['train'] == True), 'In dim']))
            y = pd.DataFrame(list(df.loc[(df['id'] == id_) & (df['train'] == True), 'Out dim']))
            # X = list(df.loc[id_train, 'In dim'])
            # y = list(df.loc[id_train, 'Out dim'])
            size_model.fit(X, y)

            df.loc[(df['id'] == id_), 'Prediction dim'] = \
                df.loc[(df['id'] == id_), 'In dim'].apply(lambda x: size_model.predict([x])[0])

            dim_pred = [[int(i[0]), int(i[1])] for i in df.loc[id_train, 'Prediction dim']]
            dim_out = [[int(i[0]), int(i[1])] for i in df.loc[id_train, 'Out dim']]

            if np.array_equal(dim_pred, dim_out):
                try:
                    df.loc[(df['id'] == id_), 'Prediction'] = \
                        df.loc[(df['id'] == id_), 'Prediction dim'].\
                            apply(lambda x: np.zeros([int(_x) for _x in x], np.int8))
                    df.loc[(df['id'] == id_), 'Type'] = "Lin_out"
                except ValueError:
                    print('Dimension Prediction ValueError on', id_)
                    df.loc[(df['id'] == id_), 'Type'] = "Custom_out"
            else:
                df.loc[(df['id'] == id_), 'Type'] = "Custom_out"
    return df


def round_grid(img, pixel1, pixel2, pixels_around=3):
    gridded = []
    for x in range(-pixels_around, pixels_around+1):
        for y in range(-pixels_around, pixels_around+1):
            if pixel1+x < 0 or pixel1+x < 0:
                gridded = gridded + [-1]
            elif pixel1+x > len(img)-1 or y+pixel2 > len(img[0])-1:
                gridded = gridded + [-1]
            else:
                gridded = gridded + [img[x+pixel1][y+pixel2]]
    return gridded


def generate_X_list(inp, pixels_around=3):
    return [round_grid(inp, i, j, pixels_around) for i in range(len(inp)) for j in range(len(inp[0]))]


def generate_XY(case, pixels_around=3, input_col="input", output_col="output"):
    X = np.concatenate(list(case[input_col].apply(lambda x: generate_X_list(x, pixels_around))))
    y = np.concatenate(case[output_col].apply(lambda x: np.concatenate(x)).reset_index(drop=True))
    X = pd.DataFrame.from_records(list(X))
    return X, np.array(y)


def resplit(lst, dim):
    range_list = [range(dim[0]*i, dim[0]*i+dim[0]) for i in range(dim[1])]
    return [list(np.array(lst)[i]) for i in range_list]


def plot_case(case, prediction_col="Prediction", input_col="input", output_col="output"):
    color_dict = {-1: [255, 255, 255],
                  0: [0, 0, 0],
                  1: [0, 116, 217],
                  2: [255, 65, 54],
                  3: [46, 204, 64],
                  4: [255, 220, 0],
                  5: [170, 170, 170],
                  6: [240, 18, 190],
                  7: [255, 133, 27],
                  8: [127, 219, 255],
                  9: [135, 12, 37],
                  True: [180, 180, 180]}  # replaced colors
    rgb_img = lambda inp: [[color_dict[p] for p in line] for line in inp]

    f, axarr = plt.subplots(len(case), 3, figsize=(7, 7))

    for i in range(len(case)):
        axarr[i, 0].imshow(rgb_img(case.reset_index().at[i, input_col]))
        axarr[i, 1].imshow(rgb_img(case.reset_index().at[i, output_col]))
        axarr[i, 2].imshow(rgb_img(case.reset_index().at[i, prediction_col]))
    f.show()


def predict_1to1_trees(filename='0d3d703e.json', printout=True):
    """
    Using just the train inputs generate a decision tree.

    Args:
        filename: filename to be trained/tested upon (default='0d3d703e.json')
        printout: display result?

    Returns:
        True if a 100% accuracy solution has been found
        False if it failed to find accurate solution
    """
    case = df[df['file'] == filename].copy()

    # list(pd.core.common.flatten(case["Prediction dim"]))
    dim = [int(i) for i in case["Prediction dim"].values[0]]

    pixels_around_to_test = dim[0]
    acc = [0] * pixels_around_to_test
    models = [0] * pixels_around_to_test

    for n, pixels_around in enumerate(range(1, dim[0] + 1)):
        X, y = generate_XY(case, pixels_around=pixels_around)

        per_element = int(len(X) / len(case))
        X_train, X_test, y_train, y_test = X.loc[range(len(X) - per_element)], X.loc[
            range(len(X) - per_element, len(X))], \
                                           y[range(len(X) - per_element)], y[range(len(X) - per_element, len(X))]

        model = tree.DecisionTreeClassifier(criterion="entropy")
        # model = lightgbm.LGBMClassifier()
        # model = svm.LinearSVR()

        model.fit(X_train, y_train)
        acc[n] = sum(y_test == model.predict(X_test)) / (dim[0] * dim[1])
        models[n] = model

    # best model
    if max(acc) == 1:
        best_n = range(1, dim[0] + 1)[acc.index(max(acc))]
        model = models[best_n - 1]

        case.loc[:, "Prediction"] = case['input'].apply(lambda x: resplit(model.predict(generate_X_list(x, best_n)), dim))
        print("100% solution found -", filename)
        if printout is True:
            plot_case(case, "Prediction")
        return True
    else:
        print("100% solution NOT found -", filename)
        return False


if __name__ == "__main__":
    df = load_data("ARC/data/training")
    # single case test
    print("Testing a single input")
    predict_1to1_trees('0d3d703e.json')

    # Loop through all eligible files
    print("Testing all eligible files for predict_1to1_trees()")
    full_preds = []
    for filename in set(df['file']):
        if (df.loc[df['file'] == filename, 'Type'].values[0] == "Same_out") and \
                (len(set([str(i) for i in df.loc[df['file'] == filename, 'Out dim'].values])) == 1):
            try:
                full_preds += [predict_1to1_trees(filename, printout=False)]
            except Exception:
                pass
    print("Correctly identified", sum(full_preds), "/", len(full_preds))  # 10 / 99 tested