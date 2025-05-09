import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import math

def axis_tick_sizes(ax,fontsize=8):
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    return ax


def load_data(csvFile):

    """
    Function to load data from a given csv file,
    returns loaded file, subject ID and block number

    """

    df = pd.read_csv(csvFile)

    Subjects = pd.unique(df['subjectID'])
    Blocks = pd.unique(df['BlockNumber'])

    return df,Subjects,Blocks


def circular_distance(angle1, angle2):
    # formula: shortest distance = Pi - abs(Pi - abs(angle1 - angle2))
    term1 = (angle1) - (angle2)
    sign_term1 = np.sign(term1)
    term2 = 180 - abs(term1)
    sign_term2 = np.sign(term2)
    sign_overall = sign_term1*sign_term2

    if (sign_overall != 0):
        shortest_distance = sign_overall * (180 - abs(term2))


    elif (sign_overall == 0):
        shortest_distance = (180 - abs(term2))

    return shortest_distance


def CircularDistance_Array(angle1,angle2):
    # formula: shortest distance = Pi - abs(Pi - abs(angle1 - angle2))

    # Calculates Circular Distance between all individual corresponding values of 2 arrays


    if len(angle1) != len(angle2):
        raise Exception("Sorry, the lengths of arrays don't match")

    else:

        Diff = np.empty(len(angle1))
        Diff[:] = np.nan
        # reset index of both arrays
        # angle1 = angle1.reset_index(drop=True)
        # angle2 = angle2.reset_index(drop=True)

        for i in range(len(angle1)):

            term1 = (angle1[i]) - (angle2[i])
            sign_term1 = np.sign(term1)
            term2 = 180 - abs(term1)
            sign_term2 = np.sign(term2)
            sign_overall = sign_term1 * sign_term2

            if (sign_overall != 0):
                shortest_distance = sign_overall * (180 - abs(term2))

            elif (sign_overall == 0):
                shortest_distance = (180 - abs(term2))

            Diff[i] = shortest_distance

    return Diff

def BoundLR(LR_array):

    # Constrain LR between 0 1nd 1

    # if (np.any(LR_array)): #check array is not empty
    if (np.size(LR_array != 0)):
        New_array = LR_array
        New_array[New_array<=0] = 0
        New_array[New_array>1] = 1

        return New_array

def BoundLR_LooseBoundaries(LR_array):

    # Constrain LR between 0 1nd 1

    # if (np.any(LR_array)): #check array is not empty
    if (np.size(LR_array != 0)):
        New_array = LR_array
        New_array[New_array<=-1] = -1
        New_array[New_array>2] = 2

        return New_array


def cm2inch(*tupl):
    """ This function convertes cm to inches

    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457

    :param tupl: Size of plot in cm
    :return: Converted image size in inches
    """

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def latex_plt(matplotlib):
    """ This function updates the matplotlib library to use Latex and changes some default plot parameters

    :param matplotlib: matplotlib instance
    :return: updated matplotlib instance
    """

    pgf_with_latex = {
        # "pgf.texsystem": "pdflatex",
        # "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": [],
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
        #"text.latex.unicode": True,
        "pgf.preamble": [
             r"\usepackage[utf8x]{inputenc}",
             r"\usepackage[T1]{fontenc}",
             r"\usepackage{cmbright}",
             ]
    }
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib


def label_subplots(f, texts, x_offset=-0.07, y_offset=0.015, fontsize=8, bold=False):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """

    # Get axes
    axes = f.get_axes()

    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1

        if (bold):
            f.text(x - x_offset, y + y_offset, l, size=fontsize,fontweight='bold')

        else:
            f.text(x - x_offset, y + y_offset, l, size=fontsize)

def label_axes(f,ax, texts, x_offset=-0.07, y_offset=0.015, fontsize=8, bold=False):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """


    # Cycle over subplots and place labels
    for a, l in zip(ax, texts):
        x = a.get_position().x0
        y = a.get_position().y1

        if (bold):
            f.text(x - x_offset, y + y_offset, l, size=fontsize,fontweight='bold')

        else:
            f.text(x - x_offset, y + y_offset, l, size=fontsize)

def center_x(cell_lower_left_x, cell_width, word_length):
    """ This function centers text along the x-axis

    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_width: Width of cell in which text appears
    :param word_length: Length of plotted word
    :return: Centered x-position
    """

    return cell_lower_left_x + (cell_width / 2.0) - (word_length / 2.0)


def center_y(cell_lower_left_y, cell_height, y0, word_height):
    """ This function centers text along the y-axis

    :param cell_lower_left_y: Lower left y-coordinate
    :param cell_height: Height of cell in which text appears
    :param y0: Lower bound of text (sometimes can be lower than cell_lower_left-y (i.e. letter y))
    :param word_height: Height of plotted word
    :return: Centered y-position
    """

    return cell_lower_left_y + ((cell_height / 2.0) - y0) - (word_height / 2.0)

def get_text_coords(f, ax, cell_lower_left_x, cell_lower_left_y, printed_word, fontsize):
    """ This function computes the length and height of a text und consideration of the font size

    :param f: Figure object
    :param ax: Axis object
    :param cell_lower_left_x: Lower left x-coordinate
    :param cell_lower_left_y: Lower left y-coordinate
    :param printed_word: Text of which length is computed
    :param fontsize: Specified font size
    :return: word_length, word_height, bbox: Computed word length and height and text coordinates
    """

    # Print text to lower left cell corner
    t = ax.text(cell_lower_left_x, cell_lower_left_y, printed_word, fontsize=fontsize)

    # Get text coordinates
    f.canvas.draw()
    # bbox = t.get_window_extent().inverse_transformed(ax.transData)
    bbox = t.get_window_extent().transformed(ax.transData.inverted())
    word_length = bbox.x1 - bbox.x0
    word_height = bbox.y1 - bbox.y0

    # Remove printed word
    t.set_visible(False)

    return word_length, word_height, bbox


def plot_opened_image_with_text(img_path,ax,x_loc,y_loc,zoom=0.10, text=None, text_x=None, text_y=None, text_kwargs=None):
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # mpl.use('cairo')
    """
        Plot an image at the desired location and optionally add text.

        :param img_path: Path to the image file
        :param ax: Axis handle
        :param x_loc: X-axis location for the image
        :param y_loc: Y-axis location for the image
        :param zoom: Zoom factor for the image
        :param text: Text to add to the plot (optional)
        :param text_x: X-axis location for the text (optional)
        :param text_y: Y-axis location for the text (optional)
        :param text_kwargs: Additional keyword arguments for the text (optional)
        :return: Axis handle
        """

    img = Image.open(img_path)
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (x_loc, y_loc), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    ab = AnnotationBbox(imagebox, (x_loc, y_loc), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Add text if provided
    if text and text_x is not None and text_y is not None:
        if text_kwargs is None:
            text_kwargs = {}
        ax.text(text_x, text_y, text,zorder=5, **text_kwargs)

    # Delete unnecessary axes
    ax.axis('off')

    return ax

def plot_image(f, img_path, cell_x0, cell_x1, cell_y0, ax, text_y_dist, text, text_pos, fontsize,
               zoom=0.2, cell_y1=np.nan):
    """ This function plots images and corresponding text for the task schematic

    :param f: Figure object
    :param img_path: Path of image
    :param cell_x0: Left x-position of area in which it is plotted centrally
    :param cell_x1: Rigth x-position of area in which it is plotted centrally
    :param cell_y0: Lower y-position of image -- if cell_y1 = nan
    :param ax: Plot axis
    :param text_y_dist: y-position distance to image
    :param text: Displayed text
    :param text_pos: Position of printed text (below vs. above)
    :param fontsize: Text font size
    :param zoom: Scale of image
    :param cell_y1: Upper x-position of area in which image is plotted (lower corresponds to cell_y0)
    :return ax, bbox: Axis object, image coordinates
    """

    # Open image
    img = Image.open(img_path)

    # Image zoom factor and axis and coordinates
    imagebox = OffsetImage(img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (cell_x0, cell_y0), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get cell width
    cell_width = cell_x1 - cell_x0
    image_x = cell_x0 + (cell_width/2)

    if not np.isnan(cell_y1):
        cell_height = cell_y1 - cell_y0
        image_y = cell_y0 + (cell_height / 2)
    else:
        image_y = cell_y0

    # Remove image and re-plot at correct coordinates
    ab.remove()
    ab = AnnotationBbox(imagebox, (image_x, image_y), xybox=None,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False)
    ax.add_artist(ab)

    # Get image coordinates
    f.canvas.draw()
    renderer = f.canvas.renderer
    # bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transAxes)
    bbox = imagebox.get_window_extent(renderer).transformed(ax.transData.inverted())

    if text_pos == 'left_below':
        # Plot text below image
        x = bbox.x0
        y = bbox.y0 - text_y_dist
    elif text_pos == 'centered_below':
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y0 - text_y_dist

    elif text_pos == 'left':
        # Plot text centrally above image
        word_length, word_height, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_height = bbox.y1 - bbox.y0
        y = center_y(bbox.y0, cell_height,0, word_height)
        x = bbox.x0 - text_y_dist

    else:
        # Plot text centrally above image
        word_length, _, _ = get_text_coords(f, ax, bbox.x0, bbox.y0, text, 6)
        cell_width = bbox.x1 - bbox.x0
        x = center_x(bbox.x0, cell_width, word_length)
        y = bbox.y1 + text_y_dist

    ax.text(x, y, text, fontsize=fontsize, color='k')

    return ax, bbox, ab


def add_text(f,imagebox,x,y,ax,text, text_kwargs=None):
    # this function adds text to the plot given bbox and ax and text
    # Plot text centrally above image

    # Get image coordinates
    f.canvas.draw()
    renderer = f.canvas.renderer
    # bbox = imagebox.get_window_extent(renderer).inverse_transformed(ax.transAxes)
    bbox = imagebox.get_window_extent(renderer).transformed(ax.transData.inverted())

    ax.text(x, y, text, zorder=5, **text_kwargs)


def MeanCentring (arr):
    New_arr = arr - np.mean(arr)

    return New_arr


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def safe_div_list(x, y):
    """ This function divides two numbers in lists and avoids division by zero

    :param x: x-values
    :param y: y-values
    :return: Result
    """

    c = np.full(len(y), np.nan)
    is_zero = y == 0
    c[is_zero] = 0.0
    c[is_zero == False] = x[is_zero==False]/y[is_zero == False]
    # c[is_zero is False] = x[is_zero is False]/y[is_zero is False] das ausprobieren.. unit test bestandne

    return c



def get_mean_voi(df_int, voi):
    """ This function computes mean estimation errors and perseveration

    :param df_int: Data frame with single-trial data
    :param voi: Variable of interest: 1 = estimation error, 2 = perseveration, 3 = motor_perseveration
    :return: mean_voi: Data frame containing mean estimation errors
    """

    if voi == 1:
        # mean estimation errors
        mean_voi = df_int.groupby(['subj_num', 'age_group',  'c_t'])['e_t'].mean()
    elif voi == 2:
        # mean perseveration frequency
        mean_voi = df_int.groupby(['subj_num', 'age_group'])['pers'].mean()
    elif voi == 3:
        # mean motor-perseveration frequency  todo: pers, not motor pers
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'edge'])['pers'].mean()
    else:
        # mean motor-perseveration frequency
        mean_voi = df_int.groupby(['subj_num', 'age_group', 'edge'])['motor_pers'].mean()

    # Reset index
    mean_voi = mean_voi.reset_index(drop=False)

    if voi == 1:
        mean_voi = mean_voi[mean_voi['c_t'] == 0]  # Drop cp trials
        mean_voi = mean_voi.reset_index(drop=False)  # Reset index

    return mean_voi


def get_stats(voi, exp, voi_name, test=1):
    """ This function computes the statistical hypothesis tests

    :param voi: Variable of interest
    :param exp: Current experiment
    :param voi_name: Name of voi
    :param test: Which test to compute. 1: Comparison between the age groups. 2: Test against zero
    :return: voi_median, voi_q1, voi_q3, p_values, stat: Median, 1st and 3rd quartile, p-values and test statistics
    """

    # Compute median, first and third quartile
    voi_median = voi.groupby(['age_group'])[voi_name].median()
    voi_q1 = voi.groupby(['age_group'])[voi_name].quantile(0.25)
    voi_q3 = voi.groupby(['age_group'])[voi_name].quantile(0.75)

    if test == 1:

        # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
        # Mann-Whitney U test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

        # Children and younger adults
        ch_ya_u, ch_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 3][voi_name], alternative='two-sided')

        # Children and older adults
        ch_oa_u, ch_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                              voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        # Younger and older adults
        ya_oa_u, ya_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 3][voi_name],
                                              voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        if exp == 1:

            # Test null hypothesis that that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_H, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 2][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Test null hypothesis that two groups have the same distribution of their voi using the nonparametric
            # Mann-Whitney U test
            # ----------------------------------------------------------------------------------------------------

            # Children and adolescents
            ch_ad_u, ch_ad_p = stats.mannwhitneyu(voi[voi['age_group'] == 1][voi_name],
                                                  voi[voi['age_group'] == 2][voi_name], alternative='two-sided')
            # Adolescents and younger adults
            ad_ya_u, ad_ya_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 3][voi_name], alternative='two-sided')
            # Adolescents and older adults
            ad_oa_u, ad_oa_p = stats.mannwhitneyu(voi[voi['age_group'] == 2][voi_name],
                                                  voi[voi['age_group'] == 4][voi_name], alternative='two-sided')

        else:

            # Test null hypothesis that that the population median of all age groups is equal using the nonparametric
            # Kruskal Wallis H test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
            kw_H, kw_p = stats.kruskal(voi[voi['age_group'] == 1][voi_name],
                                       voi[voi['age_group'] == 3][voi_name],
                                       voi[voi['age_group'] == 4][voi_name])

            # Set comparisons involving adolescents to nan
            ch_ad_p = np.nan
            ad_ya_p = np.nan
            ad_oa_p = np.nan
            ch_ad_u = np.nan
            ad_ya_u = np.nan
            ad_oa_u = np.nan

        # Save all p values
        p_values = np.array([round(kw_p, 3), round(ch_ad_p, 3), round(ch_ya_p, 3), round(ch_oa_p, 3), round(ad_ya_p, 3),
                             round(ad_oa_p, 3), round(ya_oa_p, 3)])
        # Save all test statistics
        stat = np.array([round(kw_H, 3), round(ch_ad_u, 3), round(ch_ya_u, 3), round(ch_oa_u, 3), round(ad_ya_u, 3),
                         round(ad_oa_u, 3), round(ya_oa_u, 3)])

        # Print results to console
        # -------------------------
        print('Kruskal-Wallis: H = %.3f, p = %.3f' % (round(kw_H, 3), round(kw_p, 3)))
        print('Children - adolescents: u = %.3f, p = %.3f' % (round(ch_ad_u, 3), round(ch_ad_p, 3)))
        print('Children - younger adults: u = %.3f, p = %.3f' % (round(ch_ya_u, 3), round(ch_ya_p, 3)))
        print('Children - older adults: u = %.3f, p = %.3f' % (round(ch_oa_u, 3), round(ch_oa_p, 3)))
        print('Adolescents - younger adults: u = %.3f, p = %.3f' % (round(ad_ya_u, 3), round(ad_ya_p, 3)))
        print('Adolescents - older adults: u = %.3f, p = %.3f' % (round(ad_oa_u, 3), round(ad_oa_p, 3)))
        print('Younger adults - older adults: u = %.3f, p = %.3f' % (round(ya_oa_u, 3), round(ya_oa_p, 3)))

        print('Children: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[1], 3), round(voi_q1[1], 3), round(voi_q3[1], 3)))
        if exp == 1:
            print('Adolescents: median = %.3f , IQR = (%.3f - %.3f)'
                  % (round(voi_median[2], 3), round(voi_q1[2], 3), round(voi_q3[2], 3)))
        print('Younger adults: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[3], 3), round(voi_q1[3], 3), round(voi_q3[3], 3)))
        print('Older adults: median = %.3f , IQR = (%.3f - %.3f)'
              % (round(voi_median[4], 3), round(voi_q1[4], 3), round(voi_q3[4], 3)))

    elif test == 2:

        # Test null hypothesis that the distribution of the differences between bucket and no bucket shift conditions
        # is symmetric about zero with the nonparametric Wilcoxon sign rank test
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)

        ch_stat, ch_p = stats.wilcoxon(voi[voi['age_group'] == 1][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')
        ya_stat, ya_p = stats.wilcoxon(voi[voi['age_group'] == 3][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')
        oa_stat, oa_p = stats.wilcoxon(voi[voi['age_group'] == 4][voi_name], y=None, zero_method='wilcox',
                                       correction=False, alternative='two-sided')

        # Save all p values
        p_values = np.array([round(ch_p, 3), round(ya_p, 3), round(oa_p, 3)])

        # Save all test statistics
        stat = np.array([round(ch_stat, 3), round(ya_stat, 3), round(oa_stat, 3)])

        # Print results to console
        # -------------------------
        print('Children: w = %.3f, p = %.3f' % (round(ch_stat, 3), round(ch_p, 3)))
        print('Younger adults: w = %.3f, p = %.3f' % (round(ya_stat, 3), round(ya_p, 3)))
        print('Older adults: w = %.3f, p = %.3f' % (round(oa_stat, 3), round(oa_p, 3)))

    return voi_median, voi_q1, voi_q3, p_values, stat

def medianprops():
    medianprops = dict(linestyle='-', linewidth=1, color='k')

    return medianprops

def compute_covariance_matrix(data):
    # covariance_matrix = np.cov(data, rowvar=False)
    covariance_matrix = stats.spearmanr(data)
    return covariance_matrix

def extract_common_subjects (df,factor_scores_df):
    # extract common subjects from the data frame and the subjects list
    # remove subjects with gender=3
    factor_scores_df = factor_scores_df[factor_scores_df['Gender'] != 3]

    # extract common subjects
    filtered_df = df[df['subjectID'].isin(factor_scores_df['subjectID'])]

    Subjects_df = pd.unique(filtered_df['subjectID'])

    # remove subjects if they contain 'subj' in their id
    # Create a boolean mask for elements that do not contain 'subj' or nan

    Subjects_df = np.array(
        [x for x in Subjects_df if pd.notna(x) and x != 'nan'])  # remove nan from subjects if it exists

    mask = np.array(['subj' not in id for id in Subjects_df])
    # Use the mask to filter out the unwanted elements
    Subjects_df = Subjects_df[mask]

    #Subjects_common = np.intersect1d(Subjects_df, subjects)

    return Subjects_df

def qns_factor_preprocessing(qns_totalscore, factor_scores=None, merge_both=True, drop_non_binary=True):

    '''Preprocess questionnaire and factor scores dataframes, rename columns, merge them if required
    Also drop non-binary participants if required and return the non-merged and merged dataframe'''

    # rename columns
    # if factor score is not none
    if factor_scores is not None:
        df_factor = factor_scores.rename(columns={'V1': 'subjectID'})

    df_qnstotal_subset = qns_totalscore.rename(columns={'REF':'subjectID','SD01_01':'Age','SD02':'Gender'})
    df_qnstotal_subset.loc[df_qnstotal_subset['Gender'] == 1, 'Gender'] = -1
    df_qnstotal_subset.loc[df_qnstotal_subset['Gender'] == 2, 'Gender'] = 1

    if drop_non_binary:
        # drop participants who have non-binary gender
        df_qnstotal_subset = df_qnstotal_subset[df_qnstotal_subset['Gender'] != 3]

    # merge with qns score and factor scores
    if merge_both and factor_scores is not None:
        df_merged = df_qnstotal_subset.merge(df_factor,on='subjectID')
        return df_qnstotal_subset,df_factor,df_merged

    else:
        return df_qnstotal_subset,df_factor

def sigmoid(x):
        return 1 / (1 + math.exp(-x))

def calculate_welch_ttest(df, group_col, value_col, group1, group2):

    """Perform Welch's t-test between two groups."""
    group1_data = df[df[group_col] == group1][value_col].astype('float')
    group2_data = df[df[group_col] == group2][value_col].astype('float')
    res = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    dof = welch_t_dof(group1_data, group2_data)

    return res.statistic, res.pvalue, dof


def welch_t_dof(group1, group2):
    """
    Calculate Welch's degrees of freedom for two independent samples.

    Parameters:
        group1 (array-like): Sample 1 values
        group2 (array-like): Sample 2 values

    Returns:
        float: Welch-Satterthwaite degrees of freedom
    """
    n1, n2 = len(group1), len(group2)
    s1_sq = np.var(group1, ddof=1)
    s2_sq = np.var(group2, ddof=1)

    numerator = (s1_sq / n1 + s2_sq / n2) ** 2
    denominator = ((s1_sq / n1) ** 2) / (n1 - 1) + ((s2_sq / n2) ** 2) / (n2 - 1)

    df = numerator / denominator
    return round(df,2)



def compute_median_iqr(arr):
    """This function computes median and IQR of given array"""

    median = round(np.nanmedian(arr),2)
    iqr = np.round((np.nanpercentile(arr, [25, 75])),2)

    return median, iqr

def compute_test_statistic(df, group_col, value_col, group1, group2, test='ttest_ind'):
    """Compute the test statistic (welch's t, mannU or test_rel) for two groups."""
    group1_data = df[df[group_col] == group1][value_col].astype('float')
    group2_data = df[df[group_col] == group2][value_col].astype('float')
    n1 = len(group1_data)
    n2 = len(group2_data)

    # Perform Welch's t-test
    if test == 'ttest_ind':
        res = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        dof = round(welch_t_dof(group1_data, group2_data),2)

    # Perform Mann-Whitney U test
    elif test == 'mannU':
        res = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        dof = None  # Mann-Whitney U test does not have degrees of freedom

    # Perform paired t-test
    elif test == 'ttest_rel':
        res = stats.ttest_rel(group1_data, group2_data)
        dof = len(group1_data) - 1  # Paired t-test degrees of freedom

    else:
        raise ValueError("Unsupported test type: {}".format(test))

    return res.statistic, res.pvalue, dof, n1, n2

