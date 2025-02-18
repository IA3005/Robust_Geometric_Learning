import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
import numpy as np
import moabb
import os
from moabb.datasets import Nakanishi2015,Kalunga2016,Lee2019_SSVEP
from moabb.evaluations.evaluations import WithinSessionEvaluation
from moabb.paradigms import FilterBankSSVEP

from src.classification import MDM,tWDA,tWDA_noStack
from src.featuresSSVEP import ExtendedSSVEPSignal
from src.CovariancesSSVEP import Covariances


moabb.set_log_level("info")


dataset =Kalunga2016()
dataset.interval = [0,5]
paradigm = FilterBankSSVEP()

X, labels, meta = paradigm.get_data(dataset=dataset,subjects=[dataset.subject_list[0]])
n_classes = len(np.unique(labels))
n = X.shape[2]-1
p = X.shape[1]
path0 = f"results/eeg/SSVEP/{dataset.code}"
exp=0
n_runs=4
random_seeds = [100*i for i in range(1,n_runs+1)]
n_jobs=-1
recap=[]

for random_seed in random_seeds:
    
    if not(os.path.exists(path0)):
        os.makedirs(path0)
    while os.path.exists(path0+f"/exp{exp}"):
        exp +=1
    os.makedirs(path0+f"/exp{exp}")
    path = path0+f"/exp{exp}/"    
    
    filename = path+"params.txt"
    with open(filename,'w') as f:
            f.write(f"_____{dataset.code}_____\n")
            f.write("##default withinsession (5folds)## \n")
            f.write(f"n={n}\n")
            f.write(f"p={p}\n")
            f.write(f"n_classes={n_classes}\n")
            f.write(f"scoring={paradigm.scoring}\n")
            f.write(f"random_seed={random_seed} \n")
    
    pipelines = {}
    
    
    
    classifier = "rMDM" 
    pipeline = make_pipeline(ExtendedSSVEPSignal(),
                              Covariances(estimator='scm'),
                              MDM(metric="riemann",n_jobs=n_jobs))
    pipelines[classifier] = pipeline  
    
    # classifier = "rMDM+oas" 
    # pipeline = make_pipeline(ExtendedSSVEPSignal(),
    #                           Covariances(estimator='oas'),
    #                           MDM(metric="riemann",n_jobs=n_jobs))
    # pipelines[classifier] = pipeline   

    
    # classifier = "rMDM_NoStack" 
    # pipeline = make_pipeline(ExtendedSSVEPSignal(stack=False),
    #                           Covariances(estimator="scm"),
    #                           MDM(metric="riemann",n_jobs=n_jobs))
    # pipelines[classifier] = pipeline
    
   
    classifier = "tWDA_df=10"
    pipeline = make_pipeline(ExtendedSSVEPSignal(),
                              Covariances(estimator='scm'),
                              tWDA(n=n,dfs=[10],n_jobs=n_jobs,path=None))
    pipelines[classifier] = pipeline
    
    # classifier = "tWDA_df=10_NoStack"
    # pipeline = make_pipeline(ExtendedSSVEPSignal(stack=False),
    #                           Covariances(estimator='scm'),
    #                           tWDA_noStack(n=n,dfs=[10],n_jobs=n_jobs,path=None))
    # pipelines[classifier] = pipeline
        
    classifier = "WDA"
    pipeline = make_pipeline(ExtendedSSVEPSignal(),
                              Covariances(estimator='scm'),
                              tWDA(n=n,dfs=[np.inf],n_jobs=n_jobs,path=None))
    pipelines[classifier] = pipeline
    
    # classifier = "WDA_NoStack"
    # pipeline = make_pipeline(ExtendedSSVEPSignal(stack=False),
    #                           Covariances(estimator='scm'),
    #                           tWDA_noStack(n=n,dfs=[np.inf],n_jobs=n_jobs,path=None))
    # pipelines[classifier] = pipeline
    
    # classifier = "tWDA_df_simple"
    # pipeline = make_pipeline(ExtendedSSVEPSignal(),
    #                           Covariances(estimator='scm'),
    #                           tWDA(n=n,dfs=None,df_estimation_method="kurtosis_estimation",n_jobs=n_jobs,path=None))
    # pipelines[classifier] = pipeline
    
    # classifier = "tWDA_df_simple_NoStack"
    # pipeline = make_pipeline(ExtendedSSVEPSignal(stack=False),
    #                           Covariances(estimator='scm'),
    #                           tWDA_noStack(n=n,dfs=None,df_estimation_method="kurtosis_estimation",n_jobs=n_jobs,path=None))
    # pipelines[classifier] = pipeline
    

    
    
    print(pipelines)
    
    
   
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,datasets=[dataset],
        overwrite=True,
        hdf5_path=path,
        random_state=random_seed,
        )
    
    results = evaluation.process(pipelines)
    results["seed"] = [random_seed]*len(results)
    results.to_csv(path+'results.csv')
    recap.append(results)
    
    
    
    
    #plot results
    subjects = np.unique(np.asarray(results["subject"]))
    methods = list(pipelines.keys())
    sessions = np.unique(np.asarray(results["session"]))
    n_sessions =len(sessions)
    
    for session in sessions:
        fig =plt.figure(figsize=[14, 5])
        sns.barplot(
            x="subject", y="score", hue="pipeline", 
            data=results[results["session"]==session],ci="sd",errwidth=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=6.5)
        plt.title(f"session {session} : withinsession classification with {n_classes} classes \n  n={n} p={p}")
        plt.savefig(path+f"withinsession classification session {session}.png")
        plt.show()
        
    if n_sessions>1:
        fig =plt.figure(figsize=[14, 5])
        sns.barplot(
            x="subject", y="score", hue="pipeline", 
            data=results,ci="sd",errwidth=0.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=6.5)
        plt.title(f"All sessions : withinsession classification with {n_classes} classes \n n={n} p={p}")
        #plt.ylim([0.25,0.95])
        #plt.grid(axis="y",which="both")
        plt.savefig(path+"withinsession classification for all sessions.png")
        plt.show()
    
    
    
    
    filename = path+"debrief.txt"
    with open(filename,'w') as ff:
        for subject in subjects:    
            ff.write(f"Subject {subject}:\n")
            recap_subject = results[results["subject"]==subject]
            for classifier in methods:
                res = recap_subject[recap_subject["pipeline"]==classifier]
                score = np.asarray(res["score"])
                qmin,median,qmax = np.quantile(score,q=[0.05,0.5,0.95])
                mean,std = np.mean(score),np.std(score)
                if n_sessions>1:
                    ff.write(f"{classifier} : Median = {median} +/- {qmin},{qmax} // Mean = {mean} +/- {std}\n")
                else:
                    ff.write(f"{classifier} : Median = {median} // Mean = {mean}\n")  
            ff.write(" \n")
        ff.write("For all subjects:\n") 
        for classifier in methods:   
            scores = np.asarray(results[results["pipeline"]==classifier]["score"])     
            qmin_all,median_all,qmax_all = np.quantile(scores,q=[0.05,0.5,0.95])
            mean_all,std_all = np.mean(scores),np.std(scores)
            ff.write(f"{classifier} : Median = {median_all} +/- {qmin_all},{qmax_all} // Mean = {mean_all} +/- {std_all} \n")


    
tab=pd.concat(recap,ignore_index=True)
tab.to_csv(path0+f'/recap_{n_runs}runs.csv')

mean_scores = {classifier:[] for classifier in methods}
filename = path0+f"/recap_{n_runs}runs.txt"
with open(filename,'w') as ff:
    ff.write("subject")
    for classifier in methods:
        ff.write(f",mean_score_{classifier},std_score_{classifier}")
    ff.write("\n")
    for subject in subjects:
        ff.write(f"{subject}")
        tab_subj = tab[tab["subject"]==subject]
        for classifier in methods:
            score = 100*np.asarray(list(tab_subj[tab_subj["pipeline"]==classifier]["score"]))
            ff.write(f",{np.mean(score)},{np.std(score)}")
        ff.write("\n")
        
    ff.write("mean_subject,")
    for classifier in methods:
        scores = np.asarray(list(tab[tab["pipeline"]==classifier]["score"]))
        ff.write(f",{np.mean(scores)},{np.std(scores)}")
        

fig =plt.figure(figsize=[14, 5])
sns.barplot(
    x="subject", y="score", hue="pipeline", 
    data=tab,ci="sd",errwidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=6.5)
plt.title(f"Mean over {n_runs} runs and on all sessions :\n  withinsession classification with {n_classes} classes")
#plt.ylim([0.25,0.95])
#plt.grid(axis="y",which="both")
plt.savefig(path0+f"/withinsession classification on all sessions using {n_runs} runs.png")
plt.show()


        
    
    
