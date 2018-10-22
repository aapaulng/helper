import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

def interactive_numerical_plot(df_num,df_Y):
    """Plot Interactive KDE graph. Allow user to choose variable, set xlim and save figure. df_Y only allow binary class

    Parameters
    ----------
    df_num : DataFrame
    df_Y : Series

    Returns
    -------
    None

    """
    from ipywidgets import HBox,Checkbox,FloatRangeSlider,VBox,ToggleButton,interactive_output,Dropdown
    from IPython.display import display

    def plot_num_and_save(xlimit,save_but,col,clip_box,clip_limit):
        nonlocal df_num, df_Y
        plt.close('all')

        if clip_box:
            clip_df_num = df_num.copy()
            clip_df_num.loc[clip_df_num[col]>clip_limit[1],col] = np.nan
            clip_df_num.loc[clip_df_num[col]<clip_limit[0],col] = np.nan
        else:
            clip_df_num = df_num

#         for i,col in zip(range(clip_df_num[col].shape[1]),clip_df_num[col]):
        fig,ax = plt.subplots(1,1,figsize=(10,5))
        sns.kdeplot(clip_df_num[col][df_Y == 0], label = 'label0').set_title(clip_df_num[col].name)
        sns.kdeplot(clip_df_num[col][df_Y == 1], label = 'label1')
        ax.set_xlim(xlimit[0],xlimit[1])
        plt.show()

        if save_but:
            fig.savefig('./plots/{}.png'.format(clip_df_num[col].name), bbox_inches='tight')

    xlimit = FloatRangeSlider(value = [df_num.iloc[:,1].min(),df_num.iloc[:,1].max()],min=df_num.iloc[:,1].min(),
                                              max=df_num.iloc[:,1].max(),step=(df_num.iloc[:,1].max()-df_num.iloc[:,1].min())/100,
                                              continuous_update=False,description='X_limit')
    save_but = ToggleButton(description='Save Figure')
    col = Dropdown(options=df_num.columns.tolist())
    clip_box = Checkbox(value=False,description='Clip ?')
    clip_limit = FloatRangeSlider(value = [df_num.iloc[:,1].min(),df_num.iloc[:,1].max()],min=df_num.iloc[:,1].min(),
                                              max=df_num.iloc[:,1].max(),step=(df_num.iloc[:,1].max()-df_num.iloc[:,1].min())/100,
                                              continuous_update=False,description='X_limit')


    out = interactive_output(plot_num_and_save,{
                    'xlimit' : xlimit,
                    'save_but':save_but,
                     'col' : col,
                     'clip_box':clip_box,
                     'clip_limit':clip_limit
                     })
#     save_but = Button(description='Save Fig')
    vbox1 = VBox([xlimit,save_but,col,clip_box,clip_limit])
    ui = HBox([vbox1,out])
    display(ui)

    def on_click(change):
        change['owner'].value = False

    def on_click_case(change):
        xlimit.min = df_num[change['new']].min()
        xlimit.max = df_num[change['new']].max()
        xlimit.step = (df_num[change['new']].max() - df_num[change['new']].min())/100
        xlimit.value = [df_num[change['new']].min(),df_num[change['new']].max()]
        clip_limit.min = df_num[change['new']].min()
        clip_limit.min = df_num[change['new']].min()
        clip_limit.max = df_num[change['new']].max()
        clip_limit.step = (df_num[change['new']].max() - df_num[change['new']].min())/100
        clip_limit.value = [df_num[change['new']].min(),df_num[change['new']].max()]

    save_but.observe(on_click, 'value')
    col.observe(on_click_case, 'value')

def test_plot(df_num,df_Y):
    """Looking at specially 4 cases. Dont use this module

    Parameters
    ----------
    df_num : DataFrame
    df_Y : DataFrame

    Returns
    -------
    None

    """
    from ipywidgets import HBox,Checkbox,FloatRangeSlider,VBox,ToggleButton,interactive_output,Dropdown
    from IPython.display import display

    def plot_num_and_save(case1,case2,case3,case4,xlimit,version,case,save_but,col):
        nonlocal df_num, df_Y
        plt.close('all')
        if version == 0:
    #         for i,col in zip(range(df_num[col].shape[1]),df_num[col]):
            fig,ax = plt.subplots(2,1,figsize=(10,10))
            if case1:
                sns.kdeplot(df_num[col][df_Y['CASE1'] == 0],ax=ax[0], label = 'CASE1 = {}'.format((df_Y['CASE1'] == 0).sum())).set_title(df_num[col].name)
            if case2:
                sns.kdeplot(df_num[col][df_Y['CASE2'] == 0],ax=ax[0], label = 'CASE2 = {}'.format((df_Y['CASE2'] == 0).sum()))
            if case3:
                sns.kdeplot(df_num[col][df_Y['CASE3'] == 0],ax=ax[0], label = 'CASE3 = {}'.format((df_Y['CASE3'] == 0).sum()))
            if case4:
                sns.kdeplot(df_num[col][df_Y['CASE4'] == 0],ax=ax[0], label = 'CASE4 = {}'.format((df_Y['CASE4'] == 0).sum()))
            ax[0].set_xlim(xlimit[0],xlimit[1])

            if case1:
                sns.kdeplot(df_num[col][df_Y['CASE1'] == 1],ax=ax[1], label = 'CASE1 = {}'.format((df_Y['CASE1'] == 1).sum())).set_title('label_1')
            if case2:
                sns.kdeplot(df_num[col][df_Y['CASE2'] == 1],ax=ax[1], label = 'CASE2 = {}'.format((df_Y['CASE2'] == 1).sum()))
            if case3:
                sns.kdeplot(df_num[col][df_Y['CASE3'] == 1],ax=ax[1], label = 'CASE3 = {}'.format((df_Y['CASE3'] == 1).sum()))
            if case4:
                sns.kdeplot(df_num[col][df_Y['CASE4'] == 1],ax=ax[1], label = 'CASE4 = {}'.format((df_Y['CASE4'] == 1).sum()))
            ax[1].set_xlim(xlimit[0],xlimit[1])
            plt.show()

            if save_but:
                fig.savefig('./plots/v0_{}.png'.format(df_num[col].name), bbox_inches='tight')
    #             fig.savefig('./plots/v0_{}_{}.png'.format(i,df_num[col].name), bbox_inches='tight')


        elif version == 1:
    #         for i,col in zip(range(df_num[col].shape[1]),df_num[col]):
            fig,ax = plt.subplots(1,1,figsize=(10,5))
            sns.kdeplot(df_num[col][df_Y[case] == 0], label = 'label0').set_title(df_num[col].name)
            sns.kdeplot(df_num[col][df_Y[case] == 1], label = 'label1')
            ax.set_xlim(xlimit[0],xlimit[1])
            plt.show()
            if save_but:
                fig.savefig('./plots/v1_{}.png'.format(df_num[col].name), bbox_inches='tight')
    #             fig.savefig('./plots/v1_{}_{}.png'.format(i,df_num[col].name), bbox_inches='tight')


    case1 = Checkbox(value=True,description='case1')
    case2 = Checkbox(value=True,description='case2')
    case3 = Checkbox(value=True,description='case3')
    case4 = Checkbox(value=True,description='case4')
#     xlimit = FloatRangeSlider(continuous_update=False,description='X_limit')
    xlimit = FloatRangeSlider(value = [df_num.iloc[:,1].min(),df_num.iloc[:,1].max()],min=df_num.iloc[:,1].min(),
                                              max=df_num.iloc[:,1].max(),step=(df_num.iloc[:,1].max()-df_num.iloc[:,1].min())/100,
                                              continuous_update=False,description='X_limit')
    version=Dropdown(options=[0,1])
    case = Dropdown(options=['CASE1','CASE2','CASE3','CASE4'])
    save_but = ToggleButton(description='Save Figure')
    col = Dropdown(options=df_num.columns)

    out = interactive_output(plot_num_and_save,{
                     'case1':case1,
                    'case2':case2,
                    'case3':case3,
                    'case4':case4,
                    'xlimit' : xlimit,
                     'version':version,
                     'case' : case,
                    'save_but':save_but,
                     'col' : col})
#     save_but = Button(description='Save Fig')
    vbox1 = VBox([case1,case2,case3,case4,xlimit,version,case,save_but,col])
    ui = HBox([vbox1,out])
    display(ui)

    def on_click(change):
        change['owner'].value = False

    def on_click_case(change):
        xlimit.min = df_num[change['new']].min()
        xlimit.max = df_num[change['new']].max()
        xlimit.step = (df_num[change['new']].max() - df_num[change['new']].min())/100
        xlimit.value = [df_num[change['new']].min(),df_num[change['new']].max()]

    save_but.observe(on_click, 'value')
    col.observe(on_click_case, 'value')

def drop_numerical_50percent_zero(df_num,df_Y):
    """Drop attribute with more than 50% zeros if and only if label_0 and label_1 has same percentage

    Parameters
    ----------
    df_num : DataFrame
    df_Y : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_num
    """
    for col in df_num.columns:
        if 0 in df_num[col].value_counts():
            foo = df_num[col].value_counts()[0]/sum(df_num[col].value_counts())
            if  foo > 0.5:
                ct_df_num = pd.crosstab(df_num[col], columns=df_Y)
                class_0_vc = ct_df_num[0][0]/sum(ct_df_num[0])
                class_1_vc = ct_df_num[1][0]/sum(ct_df_num[1])
                if abs(int(class_0_vc*100) - int(class_1_vc*100)) < 5:
                    print('Drop {} with {:.2f}% zeroes, class0 {:.2f}% zeroes, class1 {:.2f}% zeroes'.format(col,foo*100,class_0_vc*100,class_1_vc))
                    df_num.drop(col,axis=1,inplace=True)
                    continue
        bar = df_num[col].value_counts().max() / sum(df_num[col].value_counts())
        if bar > 0.8:
            value = df_num[col].value_counts().index[0]
            print('Drop {} with {:.2f}% of same value which is {:.2f}'.format(col,bar*100,value))
            df_num.drop(col,axis=1,inplace=True)
            continue

    return df_num

def outlier3std(df_num):
    """np.nan the outlier in 3std for numerical variable.

    Parameters
    ----------
    df_num : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_num.

    """
    for col in df_num.columns:
        print(col)
        lower_bound, upper_bound = np.percentile(df_num[col],[0.3,99.7])
        df_num.loc[df_num[col]<lower_bound,col] = np.nan
        df_num.loc[df_num[col]>upper_bound,col] = np.nan
    return df_num

def medianimpute(df_num):
    """Median Impute for numerical data.

    Parameters
    ----------
    df_num : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_num.

    """
    null_columns = df_num.isnull().sum()[df_num.isnull().sum() > 0].index.tolist()
    for col in null_columns:
        df_num[col].fillna(df_num[col].median(),inplace=True)
    return df_num

def modeimpute(df_cat):
    """Mode Impute for categorical data.

    Parameters
    ----------
    df_cat : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_cat.

    """
    null_columns = df_cat.isnull().sum()[df_cat.isnull().sum() > 0].index.tolist()
    for col in null_columns:
        df_cat[col].fillna(df_cat[col].mode()[0],inplace=True)
    return df_cat

def model_result(clf,y,X,cutoff=0.5):
    """Print Confusion Matrix, ROC_AUC, Lift and etc.

    Parameters
    ----------
    clf : Classifier
        Model
    y : DataFrame/np Array
    X : DataFrame/np Array

    Returns
    -------
    None

    """

    from sklearn.metrics import confusion_matrix,roc_auc_score
    # tn,fp,fn,tp= confusion_matrix(y,clf.predict(X)).flatten()
    tn,fp,fn,tp= confusion_matrix(y,[1 if i>= cutoff else 0 for i in clf.predict_proba(X)[:,1]]).flatten()
    print(clf.__class__)
    print()
    print(" n={:^6}   |     Prediction           ".format(tp+tn+fp+fn))
    print("____________|____0__________1___       ")
    print("            |   TN     |    FP                TNR/Spec\t\t"+ "Ratio of FP/TP = {:.2f}".format(fp/tp))
    print("        0   |  {:^6}  |  {:^6}    {:^6}    {:^6}%\t\t".format(tn, fp, tn+fp,round(tn/(tn+fp)*100, 2))+"Prevelance = {:.2f}%".format((fn+tp)/(fn+tp+fp+tn)*100))
    print("Actual      |__________|_________      \t\t\t\t"+"Accuracy = {:.2f}%".format((tn+tp)/(tn+tp+fn+fp)*100))
    print("            |   FN     |    TP                TPR/Sen/Recall\t"+"ROC AUC Score = {:.2f}".format(roc_auc_score(y,clf.predict_proba(X)[:,1])))
    print("        1   |  {:^6}  |  {:^6}    {:^6}    {:^6}%\t\t".format(fn, tp, fn+tp,round(tp/(tp+fn)*100,2))+"Lift = %.2f" % (tp/(tp+fp)/(tp+fn)*(tp+tn+fp+fn)))
    print("            |          |               ")
    print("               {:^6}    {:^6}          ".format(tn+fn, fp+tp))
    print()
    print("                NPV       PPV,Preci")
    print("               {:^6}%    {:^6}%".format(round(tn/(tn+fn)*100,2),round(tp/(tp+fp)*100,2)))

def print_cutoffpoint(clf,X_train,y_train,X_test=None,Y_test=None,cutoffs=[0.1,1.0,0.1]):
    """Print Confusion Matrix for different cut off point.

    Parameters
    ----------
    clf : classfier
        aka model.
    X_train : DataFrame
    y_train : Series
    X_test : DataFrame
    Y_test : Series
    cutoffs : list
        default : [0.1,1.0,0.1]. [min,max,step]

    Returns
    -------
    None

    """
    plot_df = pd.DataFrame(columns=['cutoff','train_lift','test_lift'])
    for cutoff in np.arange(cutoffs[0],cutoffs[1],cutoffs[2]):
        if X_train is not None and y_train is not None:
            printmd("<span style='color:red'>train, cutoff = **{}**</span>".format(round(cutoff,2)))
            model_result(clf,y_train,X_train,cutoff=cutoff)

        if X_test is not None  and y_test is not None :
            printmd("<span style='color:green'>train, cutoff = **{}**</span>".format(round(cutoff,2)))
            model_result(clf,y_test,X_test,cutoff=cutoff)

def chi2_remove_categorical(df_cat,df_Y):
    """Dummify df_cat and use scipy chi2_constigency test to select data with pvalue < 0.05

    Parameters
    ----------
    df_cat : DataFrame
        Do not put dummified dataframe
    df_Y : DataFrame

    Returns
    -------
    DataFrame
        Output

    """
    from scipy.stats import chi2_contingency
    cat_dum = pd.get_dummies(df_cat)
    for cname in cat_dum:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(df_Y,cat_dum[cname]))

        if p >= 0.05:
            print('drop {}, pvalue = {}'.format(cname,p))
            cat_dum.drop(cname,axis=1,inplace=True)

    return cat_dum

def chi2_remove_categorical_v2(df_cat,y):
    """Dummify df_cat and use sklearn chi2 test to select data with pvalue < 0.05. Faster approach compare to version_1

    Parameters
    ----------
    df_cat : DataFrame
        Do not put dummified dataframe
    df_Y : DataFrame

    Returns
    -------
    DataFrame
        Output

    """
    from sklearn.feature_selection import chi2, SelectKBest

    cat_dum = pd.get_dummies(df_cat)
    chi2_selector = SelectKBest(chi2,k=1)
    chi2_selector.fit(cat_dum,y)
    df = pd.DataFrame(list(zip(cat_dum.columns,chi2_selector.scores_,chi2_selector.pvalues_)),
                     columns = ['columns','scores','p_value'])
    df = df[df['p_value']<0.05].sort_values(by='p_value')

    print(df)

    return cat_dum[df['columns']]

def show_correlation(df_num):
    """Correlation Heatmap

    Parameters
    ----------
    df_num : DataFrame

    Returns
    -------
        null
    """
    trimask = np.zeros_like(df_num.corr(), dtype=np.bool)
    trimask[np.triu_indices_from(trimask)]=True

    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(50, 50), dpi=200)
    sns.heatmap(df_num.corr(method='pearson'),
                square=True,
                vmin=min(df_num.corr(method='pearson').values.flatten()),
                vmax=max(df_num.corr(method='pearson').values.flatten()),
                cmap=sns.color_palette("RdBu", 100),
                mask=trimask,
                annot=True,
                cbar=False)

def categorical_distplot(df_cat,y):
    """Distribution Plot of categorical variable

    Parameters
    ----------
    df_cat : DataFrame
    y : Series
    """
    for col in df_cat:
        a = pd.crosstab(df_cat[col],y)
        a = a.div(a.sum(axis=0), axis=1)
        a.plot(kind='bar',figsize=(16,6))
        plt.title(col)
        plt.show()
