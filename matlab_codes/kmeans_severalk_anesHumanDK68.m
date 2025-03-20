clc
clear all
close all


%%
load('/Users/user/Desktop/Postdoc-Codes/Cabral/Cabral/LEiDA_Psilocybin-master/Propofol_for_Rodrigo/DK68/BOLD_timeseries_Awake.mat')
load('/Users/user/Desktop/Postdoc-Codes/Cabral/Cabral/LEiDA_Psilocybin-master/Propofol_for_Rodrigo/DK68/BOLD_timeseries_Deep.mat')
load('/Users/user/Desktop/Postdoc-Codes/Cabral/Cabral/LEiDA_Psilocybin-master/Propofol_for_Rodrigo/DK68/BOLD_timeseries_Recovery.mat')
load('/Users/user/Desktop/Postdoc-Codes/Cabral/Cabral/LEiDA_Psilocybin-master/Propofol_for_Rodrigo/DK68/dk68.mat')

%%
AWCAT= horzcat(BOLD_timeseries_Awake{1, 1} ,BOLD_timeseries_Awake{2, 1},  BOLD_timeseries_Awake{3, 1},  BOLD_timeseries_Awake{4, 1}  ,  BOLD_timeseries_Awake{5, 1}  ,  BOLD_timeseries_Awake{6, 1}  ,  BOLD_timeseries_Awake{7, 1}  ,  BOLD_timeseries_Awake{8, 1}  ,  BOLD_timeseries_Awake{9, 1}  ,  BOLD_timeseries_Awake{10, 1}  ,  BOLD_timeseries_Awake{11, 1}  ,  BOLD_timeseries_Awake{12, 1}  ,  BOLD_timeseries_Awake{13, 1}  ,  BOLD_timeseries_Awake{14, 1}  ,  BOLD_timeseries_Awake{15, 1}  ,  BOLD_timeseries_Awake{16, 1}) ;
DPCAT= horzcat(BOLD_timeseries_Deep{1, 1} ,BOLD_timeseries_Deep{2, 1},  BOLD_timeseries_Deep{3, 1},  BOLD_timeseries_Deep{4, 1}  ,  BOLD_timeseries_Deep{5, 1}  ,  BOLD_timeseries_Deep{6, 1}  ,  BOLD_timeseries_Deep{7, 1}  ,  BOLD_timeseries_Deep{8, 1}  ,  BOLD_timeseries_Deep{9, 1}  ,  BOLD_timeseries_Deep{10, 1}  ,  BOLD_timeseries_Deep{11, 1}  ,  BOLD_timeseries_Deep{12, 1}  ,  BOLD_timeseries_Deep{13, 1}  ,  BOLD_timeseries_Deep{14, 1}  ,  BOLD_timeseries_Deep{15, 1}  ,  BOLD_timeseries_Deep{16, 1}) ;
RECAT= horzcat(BOLD_timeseries_Recovery{1, 1} ,BOLD_timeseries_Recovery{2, 1},  BOLD_timeseries_Recovery{3, 1},  BOLD_timeseries_Recovery{4, 1}  ,  BOLD_timeseries_Recovery{5, 1}  ,  BOLD_timeseries_Recovery{6, 1}  ,  BOLD_timeseries_Recovery{7, 1}  ,  BOLD_timeseries_Recovery{8, 1}  ,  BOLD_timeseries_Recovery{9, 1}  ,  BOLD_timeseries_Recovery{10, 1}  ,  BOLD_timeseries_Recovery{11, 1}  ,  BOLD_timeseries_Recovery{12, 1}  ,  BOLD_timeseries_Recovery{13, 1}  ,  BOLD_timeseries_Recovery{14, 1}  ,  BOLD_timeseries_Recovery{15, 1}  ,  BOLD_timeseries_Recovery{16, 1}) ;

AWCATZ=zscore(AWCAT,0,2);
DPCATZ=zscore(DPCAT,0,2);
RECATZ=zscore(RECAT,0,2);


n_ROI = 68;
n_time = 250;

%example_data = randn(n_ROI,n_time);
example_data= horzcat(AWCATZ,DPCATZ,RECATZ);

T_shift = 0;
do_filter = 0;

  
% zscore TS
TS = zscore(example_data,[],2); 
  
N = size(TS,1);
L = size(TS,2);
clear Phases timeseriedata
  
for seed=1:N
  x=demean(detrend(TS(seed,:)));
  
  xp  = [x(end:-1:1) x x(end:-1:1)]; % especular extention
  if do_filter
  filter_xp = filtfilt(bfilt2,afilt2,xp);
  else
  filter_xp = xp; %%% no filter !!
  end
  timeseriedata(seed,:) = filter_xp((size(x,2)+1):(size(x,2)*2));    % zero phase filter the data
  Xanalytic = hilbert(demean(timeseriedata(seed,:)));
  Amplitude(seed,:)=abs(Xanalytic);
  Phases(seed,:) = angle(Xanalytic);
    
  T=(T_shift+1):(size(Phases,2)-T_shift);
%  Amplitude(seed,:,i_sub)  =  abs(Xanalytic(T));
  
end  

Isubdiag = find(tril(ones(N),-1));

clear patt
 for t=1:size(T,2)
  kudata=sum(complex(cos(Phases(:,T(t))),sin(Phases(:,T(t)))))/N;
  syncdata(t)=abs(kudata);
  %%%%
  for i=1:N
    for j=1:i-1
     patt(i,j)=cos(adif(Phases(i,T(t)),Phases(j,T(t))));
    end
  end 
  pattern(:,t)=patt(Isubdiag);
  
 end
all_pattern2D = pattern(:,:)';

%subplot(4,1,1);

Y1=timeseriedata(1,1:250);
Y2=Amplitude(1,1:250);
Y3=cos(Phases(1,1:250));
Y4=Amplitude(1,1:250).*cos(Phases(1,1:250));
createfigure(Y1, Y2, Y3, Y4)



%%

N_REP = 100; % number of repetitions of the kmeans set to 500

REP = N_REP;

%opts = statset('Display','final','MaxIter',200,'UseParallel',1);
opts = statset('Display','final','MaxIter',200,'UseParallel',1); %2Display','final'
Kmeans_results={};
aux_data = all_pattern2D;
mink=3;
maxk=10;
%for k=mink:maxk 

for k=5
[cidx_Pha, ctrs_Pha,sum_D_Pha] = ...
    kmeans(aux_data, k, 'Distance','cityblock', 'Replicates',1, 'Options',opts);
    Kmeans_results{k}.cidx_Pha=cidx_Pha; % Cluster indices - numeric collumn vectos
    Kmeans_results{k}.ctrs_Pha=ctrs_Pha; % Cluster centroid locations
    Kmeans_results{k}.sum_D_Pha=sum_D_Pha; % Within-cluster sums of point-to-centroid distances
end

%%

%Evaluate Clustering performance
distM_fcd=squareform(pdist(aux_data,'cityblock'));
dunn_score=zeros(maxk,1);
for j=mink:maxk
    dunn_score(j)=dunns(j,distM_fcd,Kmeans_results{1,j}.cidx_Pha);
    disp(['Performance for ' num2str(j) ' clusters'])
end;

[~,ind_max]=max(dunn_score);

disp(['Best clustering solution: ' num2str(ind_max) ' clusters']);


%% TEST


L=size(Kmeans_results{1,5}.cidx_Pha,1);
T_shift = 9;
cn=5;
n_state=cn;
%CC=new82;
CC=sc_ctx;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)


figure
for bst = 1:n_state
        rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:4000)==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(4001:8000)==I(bst))/(L-2*T_shift);
        ratec(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(8001:size(Kmeans_results{1, 5}.cidx_Pha,1)  )==I(bst))/(L-2*T_shift);
end



sclen=250;
for s = 1:16
    for j= 1:n_state
        scrateaw(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha((s-1)*sclen+1:s*sclen)==I(j))/(sclen);
    end 
end

for s = 1:16
    for j= 1:n_state
        scratelpp(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(4000+(s-1)*sclen+1:4000+s*sclen)==I(j))/(sclen);
    end
end

for s = 1:16
    for j= 1:n_state
        scratedpp(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(8000+(s-1)*sclen+1:8000+s*sclen)==I(j))/(sclen);
    end 
end

grp = [zeros(16,1)',ones(16,1)',2*ones(16,1)'];
for i = 1:n_state
    c_1=scrateaw(:,i);
    c_2=scratelpp(:,i);
    c_3=scratedpp(:,i);
    CH(:,i) = [c_1' c_2' c_3'];
end


for i = 1:n_state
    subplot(3,n_state+1,i)
    colormap(jet)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    axis square
    title(['State ' num2str(i)])
end

subplot(3,n_state+1,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')


for k = 1:n_state
    subplot(3,n_state+1, n_state+1+k)
    boxplot(CH(:,k),grp)
end

subplot(3,n_state+1,2*n_state+3)
x = 1:n_state;
bar(x,ratea/sum(ratea))
hold on
er = errorbar(x,ratea/sum(ratea),std(scrateaw),std(scrateaw));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.4]);
ylabel('Probability Awake')
    
subplot(3,n_state+1,2*n_state+4)
x = 1:n_state;
bar(x,rateb/sum(rateb))
hold on
er = errorbar(x,rateb/sum(rateb),std(scratelpp),std(scratelpp));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.4]);
ylabel('Probability Deep Propofol')

subplot(3,n_state+1,2*n_state+5)
x = 1:n_state;
bar(x,ratec/sum(ratec))
hold on
er = errorbar(x,ratec/sum(ratec),std(scratedpp),std(scratedpp));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.4]);
ylabel('Probability Recovery')

%%

n_state=7;
cidx_Pha=Kmeans_results{1,n_state}.cidx_Pha;
ctrs_Pha=Kmeans_results{1,n_state}.ctrs_Pha;

NAW=12000;
NLP= 10500;
NDP= 10500;

%1:CTR 2:MCS 3:UWS
T1=transmat_from_seq(cidx_Pha(1:NAW),n_state); 
T2=transmat_from_seq(cidx_Pha(NAW+1:NAW+NLP),n_state);
T3=transmat_from_seq(cidx_Pha(NAW+NLP+1:size(cidx_Pha,1)),n_state);

T1=T1+0.000001; 
T2=T2+0.000001; 
T3=T3+0.000001; 


S1=T1^900;
SS1=S1(1,:);

S2=T2^900;
SS2=S2(1,:);

S3=T3^900;
SS3=S3(1,:);

EP1=0;
EP2=0;
EP3=0;
KSE1=0;
KSE2=0;
KSE3=0;
for i = 1:n_state
    for j = 1:n_state
       EP1 = EP1 + 0.5*(SS1(i)*T1(i,j)-SS1(j)*T1(j,i))*log((SS1(i)*T1(i,j))/(SS1(j)*T1(j,i)));
       EP2 = EP2 + 0.5*(SS2(i)*T2(i,j)-SS2(j)*T2(j,i))*log((SS2(i)*T2(i,j))/(SS2(j)*T2(j,i)));
       EP3 = EP3 + 0.5*(SS3(i)*T3(i,j)-SS3(j)*T3(j,i))*log((SS3(i)*T3(i,j))/(SS3(j)*T3(j,i)));
       KSE1 = KSE1 + -SS1(i)*T1(i,j)*log(T1(i,j))
       KSE2 = KSE2 + -SS2(i)*T2(i,j)*log(T2(i,j))
       KSE3 = KSE3 + -SS3(i)*T3(i,j)*log(T3(i,j))
    end
end


    
%%
load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('cocoSCmiarc.mat')
load('coco82.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});


example_data= horzcat(AWCAT,KETACAT);

load Kmeans_results_aw_keta_z
L=size(Kmeans_results{1,3}.cidx_Pha,1);
T_shift = 9;
cn=6;
n_state=cn;
%CC=new82;
CC=data_ct_82;
%CC=cocoSCmia;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)
    

for bst = 1:n_state
        
        rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:12000)==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(12001:size(Kmeans_results{1,3}.cidx_Pha,1))==I(bst))/(L-2*T_shift);

end


for i = 1:n_state
    subplot(2,n_state+1,i)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    colormap(jet)
    axis square
    title(['State ' num2str(i)])
end




subplot(2,n_state+1,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')

subplot(2,n_state+1,n_state+2)
bar(ratea/sum(ratea))
ylabel('Probability Awake')

subplot(2,n_state+1,n_state+3)
bar(rateb/sum(rateb))
ylabel('Probability Ketamine')

%subplot(2,n_state+1,n_state+4)
%y=  [ratea(1) rateb(1); ratea(2) rateb(2);ratea(3) rateb(3);ratea(4) rateb(4);ratea(5) rateb(5);ratea(6) rateb(6);ratea(7) rateb(7)]
%b = bar(y/sum(rateb));
%ylabel('Relative')



%%
clc
clear all
load('data_ts_Anesthesia_Cleaned.mat')
%load('data_ts_Anesthesia_CoCoMac_0.0025-0.05_Cleaned_Camilo.mat')
load('new82.mat')
load('coco82.mat')
load('cocoSCmiarc.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});

example_data= horzcat(AWCAT,LPPCAT,DPPCAT);

load Kmeans_results_aw_lpp_dpp_z
%load('Kmeans_results_aw_propofol_zNM.mat')
L=size(Kmeans_results{1,3}.cidx_Pha,1);
T_shift = 9;
cn=7;
n_state=cn;
%CC=new82;
CC=data_ct_82;
%CC=cocoSCmia;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    VCEN(:,i)=VA;
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)

MCORRCEN=corrcoef(VCEN);

for bst = 1:n_state
        
        rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:size(AWCAT,2))==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(size(AWCAT,2)+1:size(AWCAT,2)+size(LPPCAT,2))==I(bst))/(L-2*T_shift);
        ratec(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(size(AWCAT,2)+1+size(LPPCAT,2):size(Kmeans_results{1,3}.cidx_Pha,1))==I(bst))/(L-2*T_shift);

end


sclen=500;
for s = 1:size(ts_aw,2)
    for j= 1:n_state
        scrateaw(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha((s-1)*sclen+1:s*sclen)==I(j))/(sclen);
    end 
end

for s = 1:size(ts_lpp,2)
    for j= 1:n_state
        scratelpp(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(12000+(s-1)*sclen+1:12000+s*sclen)==I(j))/(sclen);
    end
end

for s = 1:size(ts_dpp,2)-1
    for j= 1:n_state
        scratedpp(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(22500+(s-1)*sclen+1:22500+s*sclen)==I(j))/(sclen);
    end 
end

grp = [zeros(24,1)',ones(21,1)',2*ones(22,1)'];
for i = 1:n_state
    c_1=scrateaw(:,i);
    c_2=scratelpp(:,i);
    c_3=scratedpp(:,i);
    CH(:,i) = [c_1' c_2' c_3'];
end


for i = 1:n_state
    subplot(3,n_state+1,i)
    colormap(jet)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    axis square
    title(['State ' num2str(i)])
end

subplot(3,n_state+1,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')


for k = 1:n_state
    subplot(3,n_state+1, n_state+1+k)
    boxplot(CH(:,k),grp)
end

subplot(3,n_state+1,2*n_state+3)
x = 1:n_state;
bar(x,ratea/sum(ratea))
hold on
er = errorbar(x,ratea/sum(ratea),std(scrateaw),std(scrateaw));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.6]);
ylabel('Probability Awake')
    
subplot(3,n_state+1,2*n_state+4)
x = 1:n_state;
bar(x,rateb/sum(rateb))
hold on
er = errorbar(x,rateb/sum(rateb),std(scratelpp),std(scratelpp));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.6]);
ylabel('Probability Low Propofol')

subplot(3,n_state+1,2*n_state+5)
x = 1:n_state;
bar(x,ratec/sum(ratec))
hold on
er = errorbar(x,ratec/sum(ratec),std(scratedpp),std(scratedpp));    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  
hold off
ylim([0 0.6]);
ylabel('Probability Deep Propofol')



subplot(3,n_state+1,2*n_state+6)

for i=1:24
    scatter(B, scrateaw(i,:),5,'g');
    hold on
end

hold on 

for i=1:21
    scatter(B, scratelpp(i,:),5,'b');
    hold on
end

hold on

for i=1:22
    scatter(B, scratedpp(i,:),5,'r');
    hold on
end

hold on

XR= B;
YDPPR= mean(scratedpp);
scatter(XR,YDPPR,7,'r')
line(XR,polyval(polyfit(XR,YDPPR,1),XR),'Color','r','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);

hold on 

XR= B;
YLPPR= mean(scratelpp);
scatter(XR,YLPPR,7,'b')
line(XR,polyval(polyfit(XR,YLPPR,1),XR),'Color','b','LineWidth',3)
belpp=polyfit(XR,YLPPR,1);
hold on 

XR= B;
YAWR= mean(scrateaw);
scatter(XR,YAWR,7,'g')
line(XR,polyval(polyfit(XR,YAWR,1),XR),'Color','g','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);

%%
clc;
clear all;
load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('coco82.mat')
load('cocoSCmiarc.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});

example_data= horzcat(AWCAT,KETACAT);

load Kmeans_results_aw_keta_z
L=size(Kmeans_results{1,3}.cidx_Pha,1);
T_shift = 9;
cn=7;
n_state=cn;
%CC=new82;
CC=data_ct_82;
%CC=cocoSCmia;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    VCEN(:,i)=VA;
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)

MCORRCEN=corrcoef(VCEN);

for bst = 1:n_state
       rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:12000)==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(12001:22982)==I(bst))/(L-2*T_shift);
end


sclen=500;
for s = 1:size(ts_aw,2)
    for j= 1:n_state
        scrateaw(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha((s-1)*sclen+1:s*sclen)==I(j))/(sclen);
    end 
end

for s = 1:size(ts_keta,2)-1
    for j= 1:n_state
        scrateketa(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(12000+(s-1)*sclen+1:12000+s*sclen)==I(j))/(sclen);
    end
end



grp = [zeros(24,1)',ones(21,1)'];
for i = 1:n_state
    c_1=scrateaw(:,i);
    c_2=scrateketa(:,i);
    CH(:,i) = [c_1' c_2'];
end


for i = 1:n_state
    subplot(3,n_state+1,i)
    colormap(jet)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    axis square
    title(['State ' num2str(i)])
end

subplot(3,n_state+1,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')



for k = 1:n_state
    subplot(3,n_state+1, n_state+1+k)
    boxplot(CH(:,k),grp)
end

subplot(3,n_state+1,2*n_state+3)
bar(ratea/sum(ratea))
ylabel('Probability Awake')

subplot(3,n_state+1,2*n_state+4)
bar(rateb/sum(rateb))
ylabel('Probability Ketamine')


subplot(3,n_state+1,2*n_state+5)

for i=1:24
    scatter(B, scrateaw(i,:),5,'g');
    hold on
end

hold on 

for i=1:21
    scatter(B, scrateketa(i,:),5,'b');
    hold on
end

hold on



XR= B;
YDPPR= mean(scrateketa);
scatter(XR,YDPPR,7,'r')
line(XR,polyval(polyfit(XR,YDPPR,1),XR),'Color','b','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);

hold on 


XR= B;
YAWR= mean(scrateaw);
scatter(XR,YAWR,7,'g')
line(XR,polyval(polyfit(XR,YAWR,1),XR),'Color','g','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);


%%
clc;
clear all;
load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('cocoSCmiarc.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});

example_data= horzcat(AWCAT,SELV2CAT,SELV4CAT);

load Kmeans_results_aw_selv2_selv4_z
%load('Kmeans_results_sevo2_4_zNM.mat', 'Kmeans_results')
load('coco82.mat', 'data_ct_82')


L=size(Kmeans_results{1,3}.cidx_Pha,1);
T_shift = 9;
cn=7;
n_state=cn;
%CC=new82;
%CC=cocoSCmia;
CC=data_ct_82;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)



for bst = 1:n_state
        
        rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:size(AWCAT,2))==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(size(AWCAT,2)+1:size(AWCAT,2)+size(SELV2CAT,2))==I(bst))/(L-2*T_shift);
        ratec(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(size(AWCAT,2)+1+size(SELV2CAT,2):size(Kmeans_results{1,3}.cidx_Pha,1))==I(bst))/(L-2*T_shift);

end

sclen=500;
for s = 1:size(ts_aw,2)
    for j= 1:n_state
        scrateaw(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha((s-1)*sclen+1:s*sclen)==I(j))/(sclen);
    end 
end

for s = 1:size(ts_selv2,2)
    for j= 1:n_state
        scratesev2(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(12000+(s-1)*sclen+1:12000+s*sclen)==I(j))/(sclen);
    end
end

for s = 1:size(ts_selv4,2)-1
    for j= 1:n_state
        scratesev4(s,j)=sum(Kmeans_results{1,cn}.cidx_Pha(21000+(s-1)*sclen+1:21000+s*sclen)==I(j))/(sclen);
    end 
end

grp = [zeros(24,1)',ones(18,1)',2*ones(10,1)'];
for i = 1:n_state
    c_1=scrateaw(:,i);
    c_2=scratesev2(:,i);
    c_3=scratesev4(:,i);
    CH(:,i) = [c_1' c_2' c_3'];
end


for i = 1:n_state
    subplot(3,n_state+1,i)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    colormap(jet)
    axis square
    title(['State ' num2str(i)])
end

subplot(3,n_state+1,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')

for k = 1:n_state
    subplot(3,n_state+1, n_state+1+k)
    boxplot(CH(:,k),grp)
end

subplot(3,n_state+1,2*n_state+3)
bar(ratea/sum(ratea))
ylabel('Probability Awake')

subplot(3,n_state+1,2*n_state+4)
bar(rateb/sum(rateb))
ylabel('Probability Low Sevoflurane')

subplot(3,n_state+1,2*n_state+5)
bar(ratec/sum(ratec))
ylabel('Probability Deep Sevoflurane')


subplot(3,n_state+1,2*n_state+6)

for i=1:24
    scatter(B, scrateaw(i,:),5,'g');
    hold on
end

hold on 

for i=1:18
    scatter(B, scratesev2(i,:),5,'b');
    hold on
end

hold on

for i=1:10
    scatter(B, scratesev4(i,:),5,'r');
    hold on
end

hold on

XR= B;
YDPPR= mean(scratesev4);
scatter(XR,YDPPR,7,'r')
line(XR,polyval(polyfit(XR,YDPPR,1),XR),'Color','r','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);

hold on 

XR= B;
YLPPR= mean(scratesev2);
scatter(XR,YLPPR,7,'b')
line(XR,polyval(polyfit(XR,YLPPR,1),XR),'Color','b','LineWidth',3)
belpp=polyfit(XR,YLPPR,1);
hold on 

XR= B;
YAWR= mean(scrateaw);
scatter(XR,YAWR,7,'g')
line(XR,polyval(polyfit(XR,YAWR,1),XR),'Color','g','LineWidth',3)
bedpp=polyfit(XR,YDPPR,1);
%%

clc
clear all
load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('cocoSCmiarc.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});

example_data= horzcat(AWCAT,LPPCAT, DPPCAT, KETACAT,SELV2CAT,SELV4CAT);

load('Kmeans_results_all.mat', 'Kmeans_results')
load('coco82.mat', 'data_ct_82')


L=size(Kmeans_results{1,3}.cidx_Pha,1);
T_shift = 9;
cn=7;
n_state=cn;
%CC=new82;
%CC=cocoSCmia;
CC=data_ct_82;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)



for bst = 1:n_state
        
        rate(bst) = sum(Kmeans_results{1,cn}.cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(1:12000)==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(12000+1:22500)==I(bst))/(L-2*T_shift);
        ratec(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(22500+1:34000)==I(bst))/(L-2*T_shift);
        rated(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(34000+1:45000)==I(bst))/(L-2*T_shift);
        ratee(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(45000+1:54000)==I(bst))/(L-2*T_shift);
        ratef(bst) = sum(Kmeans_results{1,cn}.cidx_Pha(54000+1:size(Kmeans_results{1,3}.cidx_Pha,1))==I(bst))/(L-2*T_shift);
end


for i = 1:n_state
    subplot(2,n_state,i)
    imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
    colormap(jet)
    axis square
    title(['State ' num2str(i)])
end

subplot(2,6,n_state+1)
imagesc(CC,[-1 1])
axis square
title('Connectome')

subplot(2,6,7)
bar(ratea/sum(ratea))
ylabel('Probability Awake')

subplot(2,6,8)
bar(rateb/sum(rateb))
ylabel('Probability Low propofol')

subplot(2,6,9)
bar(ratec/sum(ratec))
ylabel('Probability Deep propofol')

subplot(2,6,10)
bar(rated/sum(rated))
ylabel('Probability Ketamine')

subplot(2,6,11)
bar(ratee/sum(ratee))
ylabel('Probability Low Sevoflurane')

subplot(2,6,12)
bar(ratef/sum(ratef))
ylabel('Probability Deep Sevoflurane')

%%
n_state=7;
load Kmeans_results_aw_lpp_dpp
cn=n_state;

for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    VCEN(:,i)=VA;
    Isubdiag2 = find(tril(ones(n_state),-1));
end
    
MCORRCEN=corrcoef(VCEN)
VCR=MCORRCEN(Isubdiag2);
V=var(VCR)

%%

clc
clear all

%load Kmeans_results_aw_lpp_dpp_z
%load('Kmeans_results_aw_propofol_zNM.mat')
%load Kmeans_results_aw_lpp_dpp_z
%load Kmeans_results_aw_keta_z
%load Kmeans_results_aw_selv2_selv4_z
%load('Kmeans_results_all.mat')
load('Kmeans_results_aw.mat')

load('coco82.mat')

for cn = 3:8
n_state=cn;
%CC=new82;
CC=data_ct_82;
%CC=cocoSCmia;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    VCEN(:,i)=VA;
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)

for cn = cn
    for i = 1:cn
        QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
        VA=QQ(:);
        VCEN(:,i)=VA;
        Isubdiag2 = find(tril(ones(cn),-1));
        MC=corrcoef(VA,VC);
        CCA(i)=MC(1,2);
        subplot(6,10,10*(cn-3)+i)
        colormap(jet)
        imagesc(squareform(Kmeans_results{1,cn}.ctrs_Pha(I(i),:)),[-1 1])
        axis square
        title(['State ' num2str(i)])
    end
    subplot(6,10,11*(cn-3)+4)
    MCORRCEN=corrcoef(VCEN);
    VCR=MCORRCEN(Isubdiag2);
    V(cn)=var(VCR);
    imagesc(MCORRCEN)
    colorbar
    axis square
    title(['Variance ' num2str(V(cn))])
    hold on
end


end
%% Entroopia y Varianza de los patterns
load Kmeans_results_aw_lpp_dpp
load('coco82.mat')
cn=7;
n_state=cn;
CC=data_ct_82;
%CC=cocoSCmia;
VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(Kmeans_results{1,cn}.ctrs_Pha(i,:));
    VA=QQ(:);
    VCEN(:,i)=VA;
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)
ctrs = Kmeans_results{1,cn}.ctrs_Pha; 
%data = pattern;
%aux_data = pattern(:,:)';
%good_pattern = sum(abs(aux_data),2)>0;

for i =1:cn
    subplot(1,cn,i)
    imagesc(squareform(ctrs(I(i),:)),[-1 1])
    colormap(jet)
    axis square
    
    n = histcounts(ctrs(I(i),:),-1:.025:1);
    
    n = n./sum(n);
    
    H(i) = - nansum(n.*log2(n))'/log2(size(n,2));
    
    s(i) = std(ctrs(I(i),:));
    
    title({['H=' num2str(H(i))],['std=' num2str(s(i))]})
end

%%


%XR= repmat(B,[22 1])
%YR= scratedpp;
%stem(scratedpp','DisplayName','scrateaw')
%p = polyfit(XR,YR,1)
%f = polyval(p,XR);
%hold on
%plot(XR,f,'--r')
for s =1:size(scratelpp,1)
    for i=1:size(scratelpp,2)
        plp(i)=scratelpp(s,i);
    end
    Hlpp(s) = - nansum(plp.*log2(plp));
end

for s =1:size(scratedpp,1)
    for i=1:size(scratedpp,2)
        plp(i)=scratedpp(s,i);
    end
    Hdpp(s) = - nansum(plp.*log2(plp));
end

for s =1:size(scrateaw,1)
    for i=1:size(scrateaw,2)
        plp(i)=scrateaw(s,i);
    end
    Haw(s) = - nansum(plp.*log2(plp));
end

grp = [zeros(24,1)',ones(21,1)',2*ones(22,1)'];
h_1=Haw;
h_2=Hlpp;
h_3=Hdpp;
HH = [h_1 h_2 h_3];
boxplot(HH,grp)
hold on
scatter(ones(24,1),Haw,'g')
hold on
scatter(2*ones(21,1),Hlpp, 'b' )
hold on
scatter(3*ones(22,1),Hdpp, 'r')

%H = -nansum(rate.*log2(rate),2)/log2(n_state);


   
   






%  scatter(state_all(location==1)-.1,H(location==1),'x')
%  scatter(state_all(location==2),H(location==2),'o')
%  scatter(state_all(location==3)+.1,H(location==3),'s')



%%
load Kmeans_results_aw_lpp_dpp
n_states=7;
cidx=Kmeans_results{1,7}.cidx_Pha;
C = zeros(500,500);
TM = zeros(n_states,n_states,size(scrateaw,1));
%%% compute transition matrices
for j = 1:size(AWCAT,2)
    data = cidx((j*size(C,2)-(size(C,2)-1)):(j*size(C,2)));
    aux = full(sparse(data(1:(size(C,2)-1)),data(2:size(C,2)),1));
    if size(aux,2)<3
        aux(3,3)=0;
    end
    TM(:,:,j) = aux/size(C,2);
    
    for ii = 1:n_states
        for jj = 1:n_states
            TM_rand(ii,jj,j) = sum(data==ii)*sum(data==jj)/size(C,2)/size(C,2);
        end
    end
    
    
end

TMTAW = zeros(n_states,n_states);
data = cidx(1:12000);
aux = full(sparse(data(1:11999),data(2:12000),1));
for ii = 1:n_states
    TMTAW(ii,:) = aux(ii,:)/sum(aux(ii,:));
end

TMTLPP = zeros(n_states,n_states);
data = cidx(12001:22500);
aux = full(sparse(data(1:10498),data(2:10499),1));
for ii = 1:n_states
    TMTLPP(ii,:) = aux(ii,:)/sum(aux(ii,:));
end

TMTDPP = zeros(n_states,n_states);
data = cidx(22501:33855);
aux = full(sparse(data(1:11353),data(2:11354),1));
for ii = 1:n_states
    TMTDPP(ii,:) = aux(ii,:)/sum(aux(ii,:));
end

%%

clc
clear all
close all

load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('cocoSCmiarc.mat')


AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});
n_ROI = 82;
n_time = 5000;

AWCATZ = zscore(AWCAT,0,2); 
LPPCATZ = zscore(LPPCAT,0,2); 
DPPCATZ = zscore(DPPCAT,0,2); 
KETACATZ = zscore(KETACAT,0,2); 
SELV2CATZ = zscore(SELV2CAT,0,2); 
SELV4CAT = zscore(SELV4CAT,0,2); 
%example_data = randn(n_ROI,n_time);
example_data= horzcat(AWCATZ,LPPCATZ,DPPCATZ);

T_shift = 9;
do_filter = 1; %yes no filter


delta=2.4;                          % TR acquicition value for experiment
flp=0.0025;             % Low pass frecuency of filter for Monkey
fhp=0.05;               % High pass frecuency of the filter fror Monkey
k=2;    
fnq=1/(2*delta);              % Nyquist frequency
Wn=[flp/fnq fhp/fnq];         % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn); % construct the filter
  
  
% zscore TS

TS = zscore(example_data,0,2);  
  
N = size(TS,1);
L = size(TS,2);
clear Phases timeseriedata
  
for seed=1:N
  x=demean(detrend(TS(seed,:)));
  
  xp  = [x(end:-1:1) x x(end:-1:1)]; % especular extention
  if do_filter
  filter_xp = filtfilt(bfilt2,afilt2,xp);
  else
  filter_xp = xp; %%% no filter !!
  end
  timeseriedata(seed,:) = filter_xp((size(x,2)+1):(size(x,2)*2));    % zero phase filter the data
  Xanalytic = hilbert(demean(timeseriedata(seed,:)));
  Phases(seed,:) = angle(Xanalytic);
    
  T=(T_shift+1):(size(Phases,2)-T_shift);
%  Amplitude(seed,:,i_sub)  =  abs(Xanalytic(T));
  
end  

Isubdiag = find(tril(ones(N),-1));

clear patt
 for t=1:size(T,2)
  kudata=sum(complex(cos(Phases(:,T(t))),sin(Phases(:,T(t)))))/N;
  syncdata(t)=abs(kudata);
  %%%%
  for i=1:N
    for j=1:i-1
     patt(i,j)=cos(adif(Phases(i,T(t)),Phases(j,T(t))));
    end
  end 
  pattern(:,t)=patt(Isubdiag);
  
 end
    
 
 %%
 
 clc
clear all
close all

load('data_ts_Anesthesia_Cleaned.mat')
load('new82.mat')
load('cocoSCmiarc.mat')

addpath(genpath('fcn'));



AWCAT= horzcat(ts_aw{1},ts_aw{2},ts_aw{3},ts_aw{4},ts_aw{5},ts_aw{6},ts_aw{7},ts_aw{8},ts_aw{9},ts_aw{10},ts_aw{11},ts_aw{12},ts_aw{13},ts_aw{14},ts_aw{15},ts_aw{16},ts_aw{17},ts_aw{18},ts_aw{19},ts_aw{20},ts_aw{21},ts_aw{22},ts_aw{23},ts_aw{24});
LPPCAT= horzcat(ts_lpp{1},ts_lpp{2},ts_lpp{3},ts_lpp{4},ts_lpp{5},ts_lpp{6},ts_lpp{7},ts_lpp{8},ts_lpp{9},ts_lpp{10},ts_lpp{11},ts_lpp{12},ts_lpp{13},ts_lpp{14},ts_lpp{15},ts_lpp{16},ts_lpp{17},ts_lpp{18},ts_lpp{19},ts_lpp{20},ts_lpp{21});
DPPCAT= horzcat(ts_dpp{1},ts_dpp{2},ts_dpp{3},ts_dpp{4},ts_dpp{5},ts_dpp{6},ts_dpp{7},ts_dpp{8},ts_dpp{9},ts_dpp{10},ts_dpp{11},ts_dpp{12},ts_dpp{13},ts_dpp{14},ts_dpp{15},ts_dpp{16},ts_dpp{17},ts_dpp{18},ts_dpp{19},ts_dpp{20},ts_dpp{21},ts_dpp{22},ts_dpp{23});
KETACAT= horzcat(ts_keta{1},ts_keta{2},ts_keta{3},ts_keta{4},ts_keta{5},ts_keta{6},ts_keta{7},ts_keta{8},ts_keta{9},ts_keta{10},ts_keta{11},ts_keta{12},ts_keta{13},ts_keta{14},ts_keta{15},ts_keta{16},ts_keta{17},ts_keta{18},ts_keta{19},ts_keta{20},ts_keta{21},ts_keta{22});
SELV2CAT= horzcat(ts_selv2{1},ts_selv2{2},ts_selv2{3},ts_selv2{4},ts_selv2{5},ts_selv2{6},ts_selv2{7},ts_selv2{8},ts_selv2{9},ts_selv2{10},ts_selv2{11},ts_selv2{12},ts_selv2{13},ts_selv2{14},ts_selv2{15},ts_selv2{16},ts_selv2{17},ts_selv2{18});
SELV4CAT= horzcat(ts_selv4{1},ts_selv4{2},ts_selv4{3},ts_selv4{4},ts_selv4{5},ts_selv4{6},ts_selv4{7},ts_selv4{8},ts_selv4{9},ts_selv4{10},ts_selv4{11});

AWCATZ = zscore(AWCAT,0,2); 
LPPCATZ = zscore(LPPCAT,0,2); 
DPPCATZ = zscore(DPPCAT,0,2); 
KETACATZ = zscore(KETACAT,0,2); 
SELV2CATZ = zscore(SELV2CAT,0,2); 
SELV4CAT = zscore(SELV4CAT,0,2); 
%example_data = randn(n_ROI,n_time);
example_data= horzcat(AWCATZ,LPPCATZ,DPPCATZ);
%example_data= horzcat(AWCAT,SELV2CAT,SELV4CAT);
%example_data= horzcat(AWCAT,KETACAT);
%example_data=AWCAT;


T_shift = 9;
do_filter = 0; %yes no filter


delta=2.4;                          % TR acquicition value for experiment
flp=0.0025;             % Low pass frecuency of filter for Monkey
fhp=0.05;               % High pass frecuency of the filter fror Monkey
k=2;    
fnq=1/(2*delta);              % Nyquist frequency
Wn=[flp/fnq fhp/fnq];         % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn); % construct the filter
  
  
% zscore TS
%TS = zscore(example_data,[],2); 
TS = zscore(example_data,0,2); 
%TS = zscore(TS2,0,1); 
N = size(TS,1);
L = size(TS,2);
T=L;
M = N*(N - 1)/2;

  
for seed=1:N
  x=demean(detrend(TS(seed,:)));
  
  xp  = [x(end:-1:1) x x(end:-1:1)]; % especular extention
  if do_filter
  filter_xp = filtfilt(bfilt2,afilt2,xp);
  else
  filter_xp = xp; %%% no filter !!
  end
  timeseriedata(seed,:) = filter_xp((size(x,2)+1):(size(x,2)*2));    % zero phase filter the data
  Xanalytic = hilbert(demean(timeseriedata(seed,:)));
  Phases(seed,:) = angle(Xanalytic);
    
  T=(T_shift+1):(size(Phases,2)-T_shift);
%  Amplitude(seed,:,i_sub)  =  abs(Xanalytic(T));
  
end  

Isubdiag = find(tril(ones(N),-1));

clear patt
 for t=1:size(T,2)
  kudata=sum(complex(cos(Phases(:,T(t))),sin(Phases(:,T(t)))))/N;
  syncdata(t)=abs(kudata);
  %%%%
  for i=1:N
    for j=1:i-1
     patt(i,j)=cos(adif(Phases(i,T(t)),Phases(j,T(t))));
    end
  end 
  pattern(:,t)=patt(Isubdiag);
  
 end



%%



all_pattern2D = pattern(:,:)';


N_REP = 5; % number of repetitions of the kmeans set to 500

REP = N_REP;

opts = statset('Display','final','MaxIter',200,'UseParallel',1); %’Display','final'
Kmeans_results={};
aux_data = all_pattern2D;
mink=3;
maxk=10;
%for k=mink:maxk  
for k=7 
[cidx_Pha, ctrs_Pha,sum_D_Pha] = ...
    kmeans(aux_data, k, 'Distance','cityblock', 'Replicates',REP, 'Options',opts);
    Kmeans_results{k}.cidx_Pha=cidx_Pha; % Cluster indices - numeric collumn vectos
    Kmeans_results{k}.ctrs_Pha=ctrs_Pha; % Cluster centroid locations
    Kmeans_results{k}.sum_D_Pha=sum_D_Pha; % Within-cluster sums of point-to-centroid distances
end


save Kmeansbn2_results_edge_aw_lpp_dpp.mat Kmeans_results

%% Markov Analysis
TMAW=transmat_from_seq(Kmeans_results{1, 7}.cidx_Pha(1:12000),7);
TMLPP=transmat_from_seq(Kmeans_results{1, 7}.cidx_Pha(12001:22500),7);
TMDPP=transmat_from_seq(Kmeans_results{1, 7}.cidx_Pha(22501:size(Kmeans_results{1, 7}.cidx_Pha,1)),7);