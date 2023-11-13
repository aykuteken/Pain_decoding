clear all
clc
close all

% cd('/Users/aykuteken/Documents/MATLAB/MORPHINE_KE_SNIRF/');
% %rmpath(genpath('/Users/aykuteken/Documents/MATLAB/fieldtrip-20220321'));savepath;
% flist=dir;
% %% For any case, utilize the preprocessing stream GUI of HomER3
% % yBlk size : time series length x chromophore x SD couple x trials
% 
% cond_tab=readtable('/Users/aykuteken/Documents/MATLAB/MorphineOthers/condition_unblind.csv');
% flist=flist(4:end);
% k=1;
% for i=1:length(flist)
% 
%     cd(flist(i).name);
% 
%     f_subj=dir;
% 
%     f_subj(1:4)=[];
% 
%     for j=1:length(f_subj)
% 
%         if strcmp(f_subj(j).name(end-4:end),'snirf')
%             %% Loading data and adding the filename to data structure
%             disp(['Reading the file - ' f_subj(j).name]);
%             ALLDATA(i).SUBJ=str2num(f_subj(j).name(5:7));
%             subj_tab_ind=find(cond_tab.('SUBJ')==ALLDATA(i).SUBJ);
%             ALLDATA(i).data_intensity(k).run=SnirfLoad(f_subj(j).name);
%             ALLDATA(i).data_intensity(k).name =f_subj(j).name;
%             %% Convert density to optical density
%             disp(['Converting the Density to Optical Density - ' f_subj(j).name]);
%             ALLDATA(i).data_OD(k).run.data = hmrR_Intensity2OD( ALLDATA(i).data_intensity(k).run.data );
%             ALLDATA(i).data_OD(k).run.stim=ALLDATA(i).data_intensity(k).run.stim;
%             ALLDATA(i).data_OD(k).run.probe=ALLDATA(i).data_intensity(k).run.probe;
%             ALLDATA(i).data_OD(k).run.aux=ALLDATA(i).data_intensity(k).run.aux;
%             ALLDATA(i).data_OD(k).run.metaDataTags=ALLDATA(i).data_intensity(k).run.metaDataTags;
%             ALLDATA(i).data_OD(k).run.formatVersion='1.0';
%             disp(['Rejecting bad trials - ' f_subj(j).name]);
%             %% Stimulus rejection
%             %             ALLDATA(i).data_OD(k).run.stim=hmrR_StimCriteria(ALLDATA(i).data_OD(k).run.stim,1,1,[-1 1]);
%             %             ALLDATA(i).data_OD(k).run.stim=hmrR_StimRejection(ALLDATA(i).data_OD(k).run.data,ALLDATA(i).data_OD(k).run.stim,[],[],[-1 30]);
%             %% Perform Motion Artifact Correction
%             disp(['Motion Artifact Correction - ' f_subj(j).name]);
%             mlActAuto = hmrR_PruneChannels(ALLDATA(i).data_OD(k).run.data, ALLDATA(i).data_OD(k).run.probe, [], [], [1e4, 1e7], 2, [0.0, 45.0]);
%             tIncAuto = hmrR_MotionArtifact(ALLDATA(i).data_OD(k).run.data, ALLDATA(i).data_OD(k).run.probe, [], mlActAuto, [], 0.5, 1, 50, 5);
%             ALLDATA(i).data_OD(k).run.data = hmrR_MotionCorrectWavelet(ALLDATA(i).data_OD(k).run.data, [], mlActAuto, 1.5, 1);
%             ALLDATA(i).data_OD(k).run.data = hmrR_MotionCorrectPCA(ALLDATA(i).data_OD(k).run.data,[],mlActAuto,[],tIncAuto,0.97);
%             %% Perform band-pass filtering
%             disp(['Band-pass filtering - ' f_subj(j).name]);
%             [ALLDATA(i).data_OD(k).run.data, ~] = hmrR_BandpassFilt( ALLDATA(i).data_OD(k).run.data, 0.01, 0.1 );
% 
%             disp(['Optical Density to Concentration Change - ' f_subj(j).name]);
%             %% Optical Density to Concentration Change
%             ALLDATA(i).data_conc(k).run.data=hmrR_OD2Conc(ALLDATA(i).data_OD(k).run.data,ALLDATA(i).data_OD(k).run.probe,[1 1]);
%             disp(['Epoch extraction and averaging- ' f_subj(j).name]);
%             %% Epoch extraction and averaging.
%             [ALLDATA(i).data_conc(k).run.data_avg, ALLDATA(i).data_conc(k).run.data_std,ALLDATA(i).data_conc(k).run.nTrials, ALLDATA(i).data_conc(k).run.data_sum, ALLDATA(i).data_avg(k).run.yTrials] = hmrR_BlockAvg( ALLDATA(i).data_conc(k).run.data, ALLDATA(i).data_OD(k).run.stim, [-1 30] );
%             ALLDATA(i).data_conc(k).run.stim=ALLDATA(i).data_OD(k).run.stim;
%             ALLDATA(i).data_conc(k).run.probe=ALLDATA(i).data_OD(k).run.probe;
%             ALLDATA(i).data_conc(k).run.aux=ALLDATA(i).data_OD(k).run.aux;
%             ALLDATA(i).data_conc(k).run.metaDataTags=ALLDATA(i).data_OD(k).run.metaDataTags;
%             ALLDATA(i).data_conc(k).run.formatVersion='1.0';
%             ALLDATA(i).data_conc(k).run.fname = f_subj(j).name;
%             if strcmp(f_subj(j).name(end-7:end-6),'V2')
%                 if strcmp(cond_tab.('MORPHINE'){subj_tab_ind},'V2')
%                     ALLDATA(i).data_conc(k).run.Cond='MORPHINE';
%                 elseif strcmp(cond_tab.('PLACEBO'){subj_tab_ind},'V2')
%                     ALLDATA(i).data_conc(k).run.Cond='PLACEBO';
%                 end
% 
%             else %% means V1
% 
%                 if strcmp(cond_tab.('MORPHINE'){subj_tab_ind},'V1')
%                     ALLDATA(i).data_conc(k).run.Cond='MORPHINE';
%                 elseif strcmp(cond_tab.('PLACEBO'){subj_tab_ind},'V1')
%                     ALLDATA(i).data_conc(k).run.Cond='PLACEBO';
%                 end
%             end
% 
%             %             disp(['General Linear Model and SS regression analysis- ' f_subj(j).name]);
%             %             for p=1:length(ALLDATA(i).data_conc(k).run.aux)
%             %
%             %                 aux(:,p)=ALLDATA(i).data_conc(k).run.aux(p).dataTimeSeries;
%             %
%             %             end
% 
%             %stim_vec = hmrR_stim_vec();
%             %% General Linear Model and SS regression
%             %             [ALLDATA(i).data_GLM(k).data_yavg, ALLDATA(i).data_GLM(k).data_yavgstd, ...
%             %                 ALLDATA(i).data_GLM(k).nTrials, ALLDATA(i).data_GLM(k).data_ynew, ...
%             %                 ALLDATA(i).data_GLM(k).data_yresid, ALLDATA(i).data_GLM(k).data_ysum2, ...
%             %                 ALLDATA(i).data_GLM(k).beta_blks, ALLDATA(i).data_GLM(k).yR_blks, ...
%             %                 ALLDATA(i).data_GLM(k).hmrstats]=hmrR_GLM(ALLDATA(i).data_conc(k).run.data, ... %% data
%             %                 ALLDATA(i).data_conc(k).run.stim,... %% stimulus vector
%             %                 ALLDATA(i).data_conc(k).run.probe, ... %% probe info
%             %                 mlActAuto, ...                         %% mlActAuto
%             %                 aux, ...                               %% auxillary regressors
%             %                 tIncAuto, ...                          %% tIncAuto
%             %                 [], ...                                %% rcMap
%             %                 [0 30], ...                            %% time range
%             %                 1, ...                                 %% GLM solve method (1,2)
%             %                 1, ...                                 %% Idx solve basis (1,2,3,4)
%             %                 [1 1], ...                             %% params
%             %                 15, ...                                %% ssd threshold
%             %                 0, ...                                 %% flag nuisance R method (0,1,2,3)
%             %                 0, ...                                 %% drift order
%             %                 0);                                    %% contrast vector
%             %
%             %             aux=[];
%             %
%             k=k+1;
%         end
% 
%     end
%     k=1;
% 
%     cd ..
% end
% 
% ALLDATA=rmfield(ALLDATA,{'data_OD','data_intensity'});
% cd('/Users/aykuteken/Documents/MATLAB/');
% save('Morphine_Placebo.mat','ALLDATA');
load Morphine_Placebo.mat
%% SD Coupling and Subtraction
k=1;
for i=1:length(ALLDATA(1).data_conc(1).run.data.measurementList)

    if strcmp(ALLDATA(1).data_conc(1).run.data.measurementList(i).dataTypeLabel,'HbO')
        SD(k,1)=ALLDATA(1).data_conc(1).run.data.measurementList(i).sourceIndex;
        SD(k,2)=ALLDATA(1).data_conc(1).run.data.measurementList(i).detectorIndex;
        if abs(SD(k,1)-SD(k,2))>10
            chan_type{k,1}='Short';
        else
            chan_type{k,1}='Long';
        end
        k=k+1;
    end

end
%% Short channel subtraction list

subs=[];
p=1;
for i=1:9

    ind=find(SD(:,1)==i);

    for j=1:length(ind)-1

        subs(p,1)=ind(j);
        subs(p,2)=ind(end);
        p=p+1;
    end

end



for i=1:length(ALLDATA)

    %% Pre-scan (Morphine and Placebo data)
    %% Pre scan HbO - VAS3 and VAS7
    ALLDATA(i).pre_scan(1).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(1).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).pre_scan(1).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(1).run.yTrials(3).yblk(:,1,:,:),subs));
    ALLDATA(i).pre_scan(2).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(2).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).pre_scan(2).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(2).run.yTrials(3).yblk(:,1,:,:),subs));
    %% Pre scan Hb - VAS3 and VAS7

    ALLDATA(i).pre_scan(1).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(1).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).pre_scan(1).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(1).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).pre_scan(2).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(2).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).pre_scan(2).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(2).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).pre_scan(1).Cond = ALLDATA(i).data_conc(1).run.Cond;
    ALLDATA(i).pre_scan(2).Cond = ALLDATA(i).data_conc(2).run.Cond;
    %% Post Scan 30 min HbO - VAS3 and VAS7

    ALLDATA(i).post_scan(1).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(3).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(1).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(3).run.yTrials(3).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(2).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(4).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(2).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(4).run.yTrials(3).yblk(:,1,:,:),subs));

    %% Post Scan 30 min Hb - VAS3 and VAS7

    ALLDATA(i).post_scan(1).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(3).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(1).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(3).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(2).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(4).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(2).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(4).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(1).Cond = ALLDATA(i).data_conc(3).run.Cond;
    ALLDATA(i).post_scan(2).Cond = ALLDATA(i).data_conc(4).run.Cond;

    %% Post Scan 60 min HbO - VAS3 and VAS7

    ALLDATA(i).post_scan(3).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(5).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(3).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(5).run.yTrials(3).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(4).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(6).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(4).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(6).run.yTrials(3).yblk(:,1,:,:),subs));

    %% Post Scan 60 min Hb - VAS3 and VAS7

    ALLDATA(i).post_scan(3).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(5).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(3).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(5).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(4).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(6).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(4).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(6).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(3).Cond = ALLDATA(i).data_conc(5).run.Cond;
    ALLDATA(i).post_scan(4).Cond = ALLDATA(i).data_conc(6).run.Cond;

    %% Post Scan 90 min HbO - VAS3 and VAS7

    ALLDATA(i).post_scan(5).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(7).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(5).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(7).run.yTrials(3).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(6).data_HbO.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(8).run.yTrials(2).yblk(:,1,:,:),subs));
    ALLDATA(i).post_scan(6).data_HbO.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(8).run.yTrials(3).yblk(:,1,:,:),subs));

    %% Post Scan 90 min Hb - VAS3 and VAS7
    ALLDATA(i).post_scan(5).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(7).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(5).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(7).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(6).data_Hb.cond_vas3=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(8).run.yTrials(2).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(6).data_Hb.cond_vas7=Trial_Elimination(SSD_Removal(ALLDATA(i).data_avg(8).run.yTrials(3).yblk(:,2,:,:),subs));
    ALLDATA(i).post_scan(5).Cond = ALLDATA(i).data_conc(7).run.Cond;
    ALLDATA(i).post_scan(6).Cond = ALLDATA(i).data_conc(8).run.Cond;
end
fs =25;
time = [-1:1/25:30];
%% 5 sec stim 20 sec rest.
tim_range = [0,20];
indrange= find(time>=tim_range(1) & time<=tim_range(2));

%% Creating datasets

Morphine_HbO_pre.vas3=[];
Morphine_Hb_pre.vas3=[];
Placebo_HbO_pre.vas3=[];
Placebo_Hb_pre.vas3=[];

Morphine_HbO_pre.vas7=[];
Morphine_Hb_pre.vas7=[];
Placebo_HbO_pre.vas7=[];
Placebo_Hb_pre.vas7=[];

Morphine_HbO_post30.vas3=[];
Morphine_Hb_post30.vas3=[];
Placebo_HbO_post30.vas3=[];
Placebo_Hb_post30.vas3=[];

Morphine_HbO_post30.vas7=[];
Morphine_Hb_post30.vas7=[];
Placebo_HbO_post30.vas7=[];
Placebo_Hb_post30.vas7=[];

Morphine_HbO_post60.vas3=[];
Morphine_Hb_post60.vas3=[];
Placebo_HbO_post60.vas3=[];
Placebo_Hb_post60.vas3=[];

Morphine_HbO_post60.vas7=[];
Morphine_Hb_post60.vas7=[];
Placebo_HbO_post60.vas7=[];
Placebo_Hb_post60.vas7=[];

Morphine_HbO_post90.vas3=[];
Morphine_Hb_post90.vas3=[];
Placebo_HbO_post90.vas3=[];
Placebo_Hb_post90.vas3=[];

Morphine_HbO_post90.vas7=[];
Morphine_Hb_post90.vas7=[];
Placebo_HbO_post90.vas7=[];
Placebo_Hb_post90.vas7=[];

Morphine_pre.vas7_subj=[];
Placebo_pre.vas7_subj=[];
Morphine_pre.vas3_subj=[];
Placebo_pre.vas3_subj=[];

Morphine_post30.vas7_subj=[];
Placebo_post30.vas7_subj=[];
Morphine_post30.vas3_subj=[];
Placebo_post30.vas3_subj=[];

Morphine_post60.vas7_subj=[];
Placebo_post60.vas7_subj=[];
Morphine_post60.vas3_subj=[];
Placebo_post60.vas3_subj=[];

Morphine_post90.vas7_subj=[];
Placebo_post90.vas7_subj=[];
Morphine_post90.vas3_subj=[];
Placebo_post90.vas3_subj=[];



for i=1:length(ALLDATA)

    for j=1:2
        if strcmp(ALLDATA(i).pre_scan(j).Cond,'MORPHINE')
            disp(['Subject ' num2str(i) 'Pre-Scan Morphine'])
            Morphine_HbO_pre.vas3=[Morphine_HbO_pre.vas3; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas3)];
            Morphine_Hb_pre.vas3=[Morphine_Hb_pre.vas3; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_Hb.cond_vas3)];
            Morphine_HbO_pre.vas7=[Morphine_HbO_pre.vas7; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas7)];
            Morphine_Hb_pre.vas7=[Morphine_Hb_pre.vas7; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_Hb.cond_vas7)];
            Morphine_pre.vas7_subj=[Morphine_pre.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas7),1),1)*i];
            Morphine_pre.vas3_subj=[Morphine_pre.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas3),1),1)*i];
        elseif strcmp(ALLDATA(i).pre_scan(j).Cond,'PLACEBO')
            disp(['Subject ' num2str(i) 'Pre-Scan Placebo'])
            Placebo_HbO_pre.vas3=[Placebo_HbO_pre.vas3; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas3)];
            Placebo_Hb_pre.vas3=[Placebo_Hb_pre.vas3; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_Hb.cond_vas3)];
            Placebo_HbO_pre.vas7=[Placebo_HbO_pre.vas7; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas7)];
            Placebo_Hb_pre.vas7=[Placebo_Hb_pre.vas7; fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_Hb.cond_vas3)];
            Placebo_pre.vas7_subj=[Placebo_pre.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas7),1),1)*i];
            Placebo_pre.vas3_subj=[Placebo_pre.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).pre_scan(j).data_HbO.cond_vas3),1),1)*i];
        end
    end

    for k=1:6

        if k==1 || k==2 %% 30 min post
            
            if strcmp(ALLDATA(i).post_scan(k).Cond,'PLACEBO') %% Placebo
                disp(['Subject ' num2str(i) 'Post-Scan 30min Placebo'])
                Placebo_HbO_post30.vas3 =[Placebo_HbO_post30.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Placebo_Hb_post30.vas3=[Placebo_Hb_post30.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];
                
                Placebo_HbO_post30.vas7 = [Placebo_HbO_post30.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Placebo_Hb_post30.vas7=[Placebo_Hb_post30.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                
                Placebo_post30.vas7_subj=[Placebo_post30.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Placebo_post30.vas3_subj=[Placebo_post30.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];

            elseif strcmp(ALLDATA(i).post_scan(k).Cond,'MORPHINE') %% Morphine
                disp(['Subject ' num2str(i) 'Post-Scan 30min Morphine'])

                Morphine_HbO_post30.vas3 =[Morphine_HbO_post30.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Morphine_Hb_post30.vas3 =[Morphine_Hb_post30.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];
                Morphine_HbO_post30.vas7 = [Morphine_HbO_post30.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Morphine_Hb_post30.vas7=[Morphine_Hb_post30.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                
                Morphine_post30.vas7_subj=[Morphine_post30.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Morphine_post30.vas3_subj=[Morphine_post30.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];

            end

        elseif k==3 || k==4 %% 60 min post

            if strcmp(ALLDATA(i).post_scan(k).Cond,'PLACEBO') %% Placebo
                disp(['Subject ' num2str(i) 'Post-Scan 60min Placebo'])
                Placebo_HbO_post60.vas3 =[Placebo_HbO_post60.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Placebo_Hb_post60.vas3=[Placebo_Hb_post60.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];

                Placebo_HbO_post60.vas7 = [Placebo_HbO_post60.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Placebo_Hb_post60.vas7=[Placebo_Hb_post60.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                Placebo_post60.vas7_subj=[Placebo_post60.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Placebo_post60.vas3_subj=[Placebo_post60.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];


            elseif strcmp(ALLDATA(i).post_scan(k).Cond,'MORPHINE') %% Morphine

                disp(['Subject ' num2str(i) 'Post-Scan 60min Morphine'])
                Morphine_HbO_post60.vas3 =[Morphine_HbO_post60.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Morphine_Hb_post60.vas3 =[Morphine_Hb_post60.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];
                Morphine_HbO_post60.vas7 = [Morphine_HbO_post60.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Morphine_Hb_post60.vas7=[Morphine_Hb_post60.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                Morphine_post60.vas7_subj=[Morphine_post60.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Morphine_post60.vas3_subj=[Morphine_post60.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];


            end

        else

            if strcmp(ALLDATA(i).post_scan(k).Cond,'PLACEBO') %% Placebo
                disp(['Subject ' num2str(i) 'Post-Scan 90min Placebo'])
                Placebo_HbO_post90.vas3 =[Placebo_HbO_post90.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Placebo_Hb_post90.vas3=[Placebo_Hb_post90.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];

                Placebo_HbO_post90.vas7 = [Placebo_HbO_post90.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Placebo_Hb_post90.vas7=[Placebo_Hb_post90.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                Placebo_post90.vas7_subj=[Placebo_post90.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Placebo_post90.vas3_subj=[Placebo_post90.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];


            elseif strcmp(ALLDATA(i).post_scan(k).Cond,'MORPHINE') %% Morphine
                disp(['Subject ' num2str(i) 'Post-Scan 90min Morphine'])
                Morphine_HbO_post90.vas3 =[Morphine_HbO_post90.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3)];
                Morphine_Hb_post90.vas3 =[Morphine_Hb_post90.vas3; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas3)];

                Morphine_HbO_post90.vas7 = [Morphine_HbO_post90.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7)];
                Morphine_Hb_post90.vas7=[Morphine_Hb_post90.vas7; fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_Hb.cond_vas7)];
                Morphine_post90.vas7_subj=[Morphine_post90.vas7_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas7),1),1)*i];
                
                Morphine_post90.vas3_subj=[Morphine_post90.vas3_subj;ones(size(fNIRS_DS_RS(ALLDATA(i).post_scan(k).data_HbO.cond_vas3),1),1)*i];

            end
        end
    end
end
cd('/Users/aykuteken/Documents/MATLAB/');
save('Feature_fNIRS_TF.mat','Morphine_HbO_pre','Placebo_HbO_pre','Placebo_pre','Morphine_pre',...
    'Morphine_HbO_post30','Placebo_HbO_post30','Morphine_post30','Placebo_post30',...
    'Morphine_HbO_post60','Placebo_HbO_post60','Morphine_post60','Placebo_post60',...
    'Morphine_HbO_post90','Placebo_HbO_post90','Morphine_post90','Placebo_post90',...
    'Morphine_Hb_pre','Placebo_Hb_pre',...    
    'Morphine_Hb_post30','Placebo_Hb_post30',...
    'Morphine_Hb_post60','Placebo_Hb_post60',...
    'Morphine_Hb_post90','Placebo_Hb_post90');


