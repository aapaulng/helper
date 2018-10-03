import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def interactive_numerical_plot(df_X,df_Y):
    """Plot Interactive KDE graph. Allow user to choose variable, set xlim and save figure. df_Y only allow binary class

    Parameters
    ----------
    df_X : DataFrame
    df_Y : Series

    Returns
    -------
    None

    """
    from ipywidgets import HBox,Checkbox,FloatRangeSlider,VBox,ToggleButton,interactive_output,Dropdown
    from IPython.display import display

    def plot_num_and_save(xlimit,save_but,col,clip_box,clip_limit):
        nonlocal df_X, df_Y
        plt.close('all')

        if clip_box:
            clip_df_X = df_X.copy()
            clip_df_X.loc[clip_df_X[col]>clip_limit[1],col] = np.nan
            clip_df_X.loc[clip_df_X[col]<clip_limit[0],col] = np.nan
        else:
            clip_df_X = df_X

#         for i,col in zip(range(clip_df_X[col].shape[1]),clip_df_X[col]):
        fig,ax = plt.subplots(1,1,figsize=(10,5))
        sns.kdeplot(clip_df_X[col][df_Y == 0], label = 'label0').set_title(clip_df_X[col].name)
        sns.kdeplot(clip_df_X[col][df_Y == 1], label = 'label1')
        ax.set_xlim(xlimit[0],xlimit[1])
        plt.show()

        if save_but:
            fig.savefig('./plots/{}.png'.format(clip_df_X[col].name), bbox_inches='tight')

    xlimit = FloatRangeSlider(value = [df_X.iloc[:,1].min(),df_X.iloc[:,1].max()],min=df_X.iloc[:,1].min(),
                                              max=df_X.iloc[:,1].max(),step=(df_X.iloc[:,1].max()-df_X.iloc[:,1].min())/100,
                                              continuous_update=False,description='X_limit')
    save_but = ToggleButton(description='Save Figure')
    col = Dropdown(options=df_X.columns)
    clip_box = Checkbox(value=False,description='Clip ?')
    clip_limit = FloatRangeSlider(value = [df_X.iloc[:,1].min(),df_X.iloc[:,1].max()],min=df_X.iloc[:,1].min(),
                                              max=df_X.iloc[:,1].max(),step=(df_X.iloc[:,1].max()-df_X.iloc[:,1].min())/100,
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
        xlimit.min = df_X[change['new']].min()
        xlimit.max = df_X[change['new']].max()
        xlimit.step = (df_X[change['new']].max() - df_X[change['new']].min())/100
        xlimit.value = [df_X[change['new']].min(),df_X[change['new']].max()]
        clip_limit.min = df_X[change['new']].min()
        clip_limit.min = df_X[change['new']].min()
        clip_limit.max = df_X[change['new']].max()
        clip_limit.step = (df_X[change['new']].max() - df_X[change['new']].min())/100
        clip_limit.value = [df_X[change['new']].min(),df_X[change['new']].max()]

    save_but.observe(on_click, 'value')
    col.observe(on_click_case, 'value')

def test_plot(df_X,df_Y):
    """Looking at specially 4 cases. Dont use this module

    Parameters
    ----------
    df_X : DataFrame
    df_Y : DataFrame

    Returns
    -------
    None

    """
    from ipywidgets import HBox,Checkbox,FloatRangeSlider,VBox,ToggleButton,interactive_output,Dropdown
    from IPython.display import display

    def plot_num_and_save(case1,case2,case3,case4,xlimit,version,case,save_but,col):
        nonlocal df_X, df_Y
        plt.close('all')
        if version == 0:
    #         for i,col in zip(range(df_X[col].shape[1]),df_X[col]):
            fig,ax = plt.subplots(2,1,figsize=(10,10))
            if case1:
                sns.kdeplot(df_X[col][df_Y['CASE1'] == 0],ax=ax[0], label = 'CASE1 = {}'.format((df_Y['CASE1'] == 0).sum())).set_title(df_X[col].name)
            if case2:
                sns.kdeplot(df_X[col][df_Y['CASE2'] == 0],ax=ax[0], label = 'CASE2 = {}'.format((df_Y['CASE2'] == 0).sum()))
            if case3:
                sns.kdeplot(df_X[col][df_Y['CASE3'] == 0],ax=ax[0], label = 'CASE3 = {}'.format((df_Y['CASE3'] == 0).sum()))
            if case4:
                sns.kdeplot(df_X[col][df_Y['CASE4'] == 0],ax=ax[0], label = 'CASE4 = {}'.format((df_Y['CASE4'] == 0).sum()))
            ax[0].set_xlim(xlimit[0],xlimit[1])

            if case1:
                sns.kdeplot(df_X[col][df_Y['CASE1'] == 1],ax=ax[1], label = 'CASE1 = {}'.format((df_Y['CASE1'] == 1).sum())).set_title('label_1')
            if case2:
                sns.kdeplot(df_X[col][df_Y['CASE2'] == 1],ax=ax[1], label = 'CASE2 = {}'.format((df_Y['CASE2'] == 1).sum()))
            if case3:
                sns.kdeplot(df_X[col][df_Y['CASE3'] == 1],ax=ax[1], label = 'CASE3 = {}'.format((df_Y['CASE3'] == 1).sum()))
            if case4:
                sns.kdeplot(df_X[col][df_Y['CASE4'] == 1],ax=ax[1], label = 'CASE4 = {}'.format((df_Y['CASE4'] == 1).sum()))
            ax[1].set_xlim(xlimit[0],xlimit[1])
            plt.show()

            if save_but:
                fig.savefig('./plots/v0_{}.png'.format(df_X[col].name), bbox_inches='tight')
    #             fig.savefig('./plots/v0_{}_{}.png'.format(i,df_X[col].name), bbox_inches='tight')


        elif version == 1:
    #         for i,col in zip(range(df_X[col].shape[1]),df_X[col]):
            fig,ax = plt.subplots(1,1,figsize=(10,5))
            sns.kdeplot(df_X[col][df_Y[case] == 0], label = 'label0').set_title(df_X[col].name)
            sns.kdeplot(df_X[col][df_Y[case] == 1], label = 'label1')
            ax.set_xlim(xlimit[0],xlimit[1])
            plt.show()
            if save_but:
                fig.savefig('./plots/v1_{}.png'.format(df_X[col].name), bbox_inches='tight')
    #             fig.savefig('./plots/v1_{}_{}.png'.format(i,df_X[col].name), bbox_inches='tight')


    case1 = Checkbox(value=True,description='case1')
    case2 = Checkbox(value=True,description='case2')
    case3 = Checkbox(value=True,description='case3')
    case4 = Checkbox(value=True,description='case4')
#     xlimit = FloatRangeSlider(continuous_update=False,description='X_limit')
    xlimit = FloatRangeSlider(value = [df_X.iloc[:,1].min(),df_X.iloc[:,1].max()],min=df_X.iloc[:,1].min(),
                                              max=df_X.iloc[:,1].max(),step=(df_X.iloc[:,1].max()-df_X.iloc[:,1].min())/100,
                                              continuous_update=False,description='X_limit')
    version=Dropdown(options=[0,1])
    case = Dropdown(options=['CASE1','CASE2','CASE3','CASE4'])
    save_but = ToggleButton(description='Save Figure')
    col = Dropdown(options=df_X.columns)

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
        xlimit.min = df_X[change['new']].min()
        xlimit.max = df_X[change['new']].max()
        xlimit.step = (df_X[change['new']].max() - df_X[change['new']].min())/100
        xlimit.value = [df_X[change['new']].min(),df_X[change['new']].max()]

    save_but.observe(on_click, 'value')
    col.observe(on_click_case, 'value')

def drop_numerical_50percent_zero(df_X,df_Y):
    """Drop attribute with more than 50% zeros if and only if label_0 and label_1 has same percentage

    Parameters
    ----------
    df_X : DataFrame
    df_Y : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_X
    """
    for col in df_X.columns:
        if 0 in df_X[col].value_counts():
            foo = df_X[col].value_counts()[0]/sum(df_X[col].value_counts())
            if  foo > 0.5:
                ct_df_X = pd.crosstab(df_X[col], columns=df_Y)
                class_0_vc = ct_df_X[0][0]/sum(ct_df_X[0])
                class_1_vc = ct_df_X[1][0]/sum(ct_df_X[1])
                if abs(int(class_0_vc*100) - int(class_1_vc*100)) < 5:
                    print('Drop {} with {:.2f}% zeroes, class0 {:.2f}% zeroes, class1 {:.2f}% zeroes'.format(col,foo*100,class_0_vc*100,class_1_vc))
                    df_X.drop(col,axis=1,inplace=True)
                    continue
        bar = df_X[col].value_counts().max() / sum(df_X[col].value_counts())
        if bar > 0.8:
            value = df_X[col].value_counts().index[0]
            print('Drop {} with {:.2f}% of same value which is {:.2f}'.format(col,bar*100,value))
            df_X.drop(col,axis=1,inplace=True)
            continue

    return df_X

def outlier3std(df_X):
    """np.nan the outlier in 3std.

    Parameters
    ----------
    df_X : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_X.

    """
    for col in df_X.columns:
        print(col)
        lower_bound, upper_bound = np.percentile(df_X[col],[0.3,99.7])
        df_X.loc[df_X[col]<lower_bound,col] = np.nan
        df_X.loc[df_X[col]>upper_bound,col] = np.nan
    return df_X

def medianimpute(df_X):
    """Median Impute for numerical data.

    Parameters
    ----------
    df_X : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_X.

    """
    null_columns = df_X.isnull().sum()[df_X.isnull().sum() > 0].index.tolist()
    for col in null_columns:
        df_X[col].fillna(df_X[col].median(),inplace=True)
    return df_X

def modeimpute(df_X):
    """Mode Impute for categorical data.

    Parameters
    ----------
    df_X : DataFrame

    Returns
    -------
    DataFrame
        Return modified df_X.

    """
    null_columns = df_X.isnull().sum()[df_X.isnull().sum() > 0].index.tolist()
    for col in null_columns:
        df_X[col].fillna(df_X[col].mode()[0],inplace=True)
    return df_X

def print_cutoffpoint(clf,X_train,y_train,X_test,Y_test,cutoffs=[0.1,1.0,0.1]):
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
    for cutoff in range(cutoffs):
        if X_train and y_train:
            plt.title('{:.2f} train'.format(cutoff))
            cm = confusion_matrix(y_train,[1 if x> cutoff else 0 for x in clf.predict_proba(X_train)[:,1]])
            # cm = confusion_matrix(y_train,cutoff_predict(clf,X_train,threshold))
            sns.heatmap(cm,annot=True,fmt='d')
            plt.show()

            print('train set roc_auc_score {:.2f}'.format(roc_auc_score(y_train,clf.predict_proba(X_train)[:,1])))
            print('train set  f1_score {:.2f}'.format(f1_score(y_train.reshape(-1,1),[1 if x >cutoff else 0 for x in clf.predict_proba(X_train)[:,1]])))
            train_lift = lift_score(y_train,[1 if x >cutoff else 0 for x in clf.predict_proba(X_train)[:,1]])
            print('train lift {:.2f}'.format(train_lift))

        if X_test and y_test:
            plt.title('{:.2f} test'.format(cutoff))
            cm2 = confusion_matrix(y_test,[1 if x >cutoff else 0 for x in  clf.predict_proba(X_test)[:,1]])
            # cm2 = confusion_matrix(y_test,cutoff_predict(clf,X_test,threshold))
            sns.heatmap(cm2,annot=True,fmt='d')
            plt.show()

            print('test  set  roc_auc_score {:.2f}'.format(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])))
            print('test  set  f1_score {:.2f}'.format(f1_score(y_test,[1 if x >cutoff else 0 for x in clf.predict_proba(X_test)[:,1]])))
            test_lift = lift_score(y_test,[1 if x >cutoff else 0 for x in clf.predict_proba(X_test)[:,1]])
            print('test lift {:.2f}'.format(test_lift))

def chi2_remove_categorical(df_cat,df_Y):
    from scipy.stats import chi2_contingency
    for cname in df_cat:
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(df_Y,df_cat[cname]))

        if p < 0.05:
            print('drop {}, pvalue = {}'.format(cname,p))
            df_cat.drop(cname,axis=1,inplace=True)

    return df_cat


def model_results(clf,y,X):
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
    tn,fp,fn,tp= confusion_matrix(y,clf.predict(X)).flatten()
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
