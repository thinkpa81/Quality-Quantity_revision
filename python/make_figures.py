"""Reproducible publication figures from aggregate outputs and private score arrays."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import ROOT,RES,PRIV
F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'axes.titlesize':13,'figure.facecolor':'white'})
COLORS=['#176B87','#BE5A38','#6B5B95','#39846C','#444444']

def finish(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def design():
    fig=plt.figure(figsize=(10,7));gs=fig.add_gridspec(2,1,height_ratios=[1.1,1],hspace=.4);a=fig.add_subplot(gs[0]);a.axis('off')
    boxes=[(.02,.64,.28,.27,'Raw person panel\n417,823 person-waves\n37,442 people'),(.36,.64,.28,.27,'Wage-employed baselines\n160,102 rows\nWaves 1–27'),(.70,.64,.28,.27,'Eligible baselines\n150,546 rows\nWaves 1–26'),(.36,.10,.28,.27,'Administrative end\n9,556 rows at wave 27\nNo wave-28 outcome'),(.70,.10,.28,.27,'Observed follow-ups\n139,566 rows\n19,339 people')]
    for x,y,w,h,txt in boxes:
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc='#EEF3F6',ec='#7B8B96',lw=1,transform=a.transAxes));a.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=12,transform=a.transAxes)
    for start,end in [((.30,.775),(.35,.775)),((.64,.775),(.69,.775)),((.50,.63),(.50,.39)),((.84,.63),(.84,.39))]:a.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#455A64',lw=1.3))
    a.text(.83,.48,'Excluded: 10,980\nno next-wave outcome',ha='right',va='center',fontsize=11,transform=a.transAxes,color='#9D3E27')
    a.set_title('A  Risk-set reconstruction',loc='left',pad=10)
    b=fig.add_subplot(gs[1]);rows=[('Training baselines',1,20,COLORS[0]),('Validation baselines',21,23,COLORS[1]),('Test baselines',24,26,COLORS[2])]
    for i,(lab,st,en,col) in enumerate(rows):
        b.broken_barh([(st-.4,en-st+.8)],(2-i-.22,.44),facecolors=col);b.plot(en+1,2-i,'o',mfc='white',mec=col,ms=7);b.text((st+en)/2,2-i,f'{st}–{en}',ha='center',va='center',color='white',fontsize=12)
    b.set_yticks([2,1,0],[x[0] for x in rows]);b.set_xticks([1,7,14,20,21,23,24,26,27]);b.set_xlim(.3,27.7);b.set_ylim(-.65,2.55);b.set_xlabel('Baseline survey wave (year = wave + 1997)');b.grid(axis='x',alpha=.15);b.set_title('B  One-wave horizon; open circles show latest outcome wave',loc='left',pad=10)
    finish(fig,'Figure_1_design')

def temporal():
    d=pd.read_csv(RES/'test_wave_metrics.csv');ci=pd.read_csv(RES/'per_wave_cluster_confidence_intervals.csv');fig,axs=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for name,col in [('catboost',COLORS[0]),('forward_stack',COLORS[1])]:
        x=d[(d.model==name)&(d.method=='raw')].sort_values('wave');lab='CatBoost' if name=='catboost' else 'Forward stack'
        c=ci[(ci.model==name)&(ci.metric=='roc_auc')].sort_values('wave');axs[0].errorbar(x.wave,x.roc_auc,yerr=[x.roc_auc.to_numpy()-c.lower.to_numpy(),c.upper.to_numpy()-x.roc_auc.to_numpy()],label=lab,color=col,marker='o',capsize=3)
        for method,style in [('raw','--'),('platt','-')]:
            z=d[(d.model==name)&(d.method==method)].sort_values('wave');axs[1].plot(z.wave,z.calibration_intercept,style+'o',color=col,label=lab+' '+method.capitalize());axs[2].plot(z.wave,z.brier,style+'o',color=col)
    axs[0].set_ylabel('ROC-AUC (95% cluster interval)');axs[1].set_ylabel('Calibration-in-the-large');axs[1].axhline(0,color='gray',lw=.8);axs[2].set_ylabel('Brier score')
    for i,a in enumerate(axs):a.set_xticks([24,25,26],['24\n2021','25\n2022','26\n2023']);a.set_xlabel('Baseline wave / year');a.grid(alpha=.15);a.set_title(chr(65+i),loc='left')
    handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=11)
    finish(fig,'Figure_2_temporal')

def decision():
    d=pd.read_csv(RES/'decision_curve.csv');fig,ax=plt.subplots(figsize=(8,4.5))
    for i,(name,g) in enumerate(d.groupby('model',sort=False)):
        if name=='Treat all':ax.plot(g.threshold,g.net_benefit,color='#777777',ls='--',label=name,lw=1)
        elif name=='Treat none':ax.plot(g.threshold,g.net_benefit,color='black',ls=':',label=name,lw=1)
        else:ax.plot(g.threshold,g.net_benefit,color=COLORS[i],label=name,lw=1.7)
    ax.set_ylim(-.015,.085);ax.set_xlim(.01,.5);ax.set_xlabel('Threshold probability');ax.set_ylabel('Net benefit per scored person-wave');ax.legend(frameon=False,fontsize=10,ncol=2);ax.grid(alpha=.15)
    finish(fig,'Figure_3_decision')

def calibration():
    dest=RES/'calibration_curve_bins.csv'
    if dest.exists():allbins=pd.read_csv(dest)
    else:
        t=pd.read_pickle(PRIV/'test_predictions.pkl');pieces=[]
        for name in ['logistic','catboost','forward_stack']:
            for method in ['raw','platt','isotonic']:
                p=t[name if method=='raw' else name+'_'+method].to_numpy();y=t.exit_label_t1.to_numpy();q=pd.qcut(pd.Series(p),q=10,duplicates='drop')
                bins=pd.DataFrame({'p':p,'y':y,'q':q}).groupby('q',observed=True).agg(p=('p','mean'),y=('y','mean'),n=('y','size')).reset_index(drop=True);bins['model']=name;bins['method']=method;pieces.append(bins)
        allbins=pd.concat(pieces,ignore_index=True);allbins.to_csv(dest,index=False)
    fig,axs=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
    for ax,name,label in zip(axs,['logistic','catboost','forward_stack'],['Logistic','CatBoost','Forward stack']):
        for method,col in zip(['raw','platt','isotonic'],COLORS):
            bins=allbins[(allbins.model==name)&(allbins.method==method)];ax.plot(bins.p,bins.y,'o-',color=col,label=method.capitalize(),ms=3)
        ax.plot([0,.35],[0,.35],color='gray',ls='--',lw=1);ax.set_xlim(0,.35);ax.set_ylim(0,.35);ax.set_title(label);ax.set_xlabel('Mean predicted probability');ax.grid(alpha=.15)
    axs[0].set_ylabel('Observed event rate');axs[-1].legend(frameon=False,fontsize=11);finish(fig,'Figure_S1_calibration')

def shap():
    d=pd.read_csv(RES/'shap_primary_importance.csv').head(20).iloc[::-1];fig,ax=plt.subplots(figsize=(8,6));ax.barh(d.feature,d.mean_abs_shap,color=COLORS[0]);ax.set_xlabel('Mean absolute SHAP value (raw log-odds scale)');ax.grid(axis='x',alpha=.15);finish(fig,'Figure_S2_shap')

if __name__=='__main__':
    design();temporal();decision();calibration();shap();print('FIGURES COMPLETE')
