clc
clear all
close all


load('LSD_TS_FC.mat', 'DataCorrel')
load('Structural.mat')

%%
LSDACAT= horzcat(DataCorrel(1).LSD_TS,DataCorrel(2).LSD_TS,DataCorrel(3).LSD_TS,DataCorrel(4).LSD_TS,DataCorrel(5).LSD_TS,DataCorrel(6).LSD_TS,DataCorrel(7).LSD_TS,DataCorrel(8).LSD_TS,DataCorrel(9).LSD_TS,DataCorrel(10).LSD_TS,DataCorrel(11).LSD_TS,DataCorrel(12).LSD_TS,DataCorrel(13).LSD_TS,DataCorrel(14).LSD_TS,DataCorrel(15).LSD_TS,DataCorrel(16).LSD_TS);
LSDPCAT= horzcat(DataCorrel(1).PCB_TS,DataCorrel(2).PCB_TS ,DataCorrel(3).PCB_TS ,DataCorrel(4).PCB_TS ,DataCorrel(5).PCB_TS ,DataCorrel(6).PCB_TS ,DataCorrel(7).PCB_TS ,DataCorrel(8).PCB_TS ,DataCorrel(9).PCB_TS ,DataCorrel(10).PCB_TS ,DataCorrel(11).PCB_TS ,DataCorrel(12).PCB_TS ,DataCorrel(13).PCB_TS ,DataCorrel(14).PCB_TS ,DataCorrel(15).PCB_TS,DataCorrel(16).PCB_TS );
example_data= horzcat(LSDACAT,LSDPCAT);
%example_data= horzcat(PLA5CAT);


T_shift = 9;
do_filter = 0; %yes no filter


TR=2;

% Bandpass filter settings              
fnq=1/(2*TR);                 % Nyquist frequency
flp = .01;                    % lowpass frequency of filter
fhi = 0.1;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt2,afilt2]=butter(k,Wn);   % construct the filter
  
  
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
%%
%%% remove empty images
all_pattern2D = pattern(:,:)';
disp('size before removing empty images')
size(all_pattern2D)
good_pattern = sum(abs(all_pattern2D),2)>0;
all_pattern2D = all_pattern2D(good_pattern,:);
disp('size after removing empty images')

size(all_pattern2D)

%%% remove outliers
D = squareform(pdist(all_pattern2D,'cityblock'));
D = zscore(mean(D));
good_pattern = D<3;
all_pattern2D = all_pattern2D(good_pattern,:);
disp('size after removing outliers')

size(all_pattern2D)
%%
% kmeans clustering
%load('LSDnew.mat')
%load('Structural.mat')
%load('startl.mat')
%load('startl9.mat')
%load('pla5_all_pattern2D.mat')
N_REP = 5; % number of repetitions of the kmeans set to 500
for n_state = 5
REP = N_REP;
opts = statset('Display','final','MaxIter',100,'UseParallel',1);
%opts = statset('Display','final','MaxIter',200,'UseParallel',1); %'Display','final'
aux_data = all_pattern2D;
[cidx_Pha, ctrs_Pha,sum_D_Pha] = ...
    kmeans(aux_data, n_state, 'Distance','cityblock', 'Replicates',1, 'Options',opts);
end


%%
%CC=new82;

n_state=5;
CC=SC;

% Define id_aal and id_deco
id_aal = linspace(1, 90, 90);
id_deco = [linspace(1, 89, 45) fliplr(linspace(2, 90, 45))];

% Example matrix with values (replace this with your actual matrix)
matrix_values = CC; % Example 90x90 matrix with random values

% Create a mapping from id_deco to id_aal
[~, deco_to_aal_mapping] = ismember(id_deco, id_aal);

% Reorder the matrix using the mapping
CC2= matrix_values(deco_to_aal_mapping, deco_to_aal_mapping);


VC2=CC(:);
VC=im2double(VC2);
CCA=zeros(1,n_state);
for i = 1:n_state
    QQ=squareform(ctrs_Pha(i,:));
    VA=QQ(:);
    MC=corrcoef(VA,VC);
    CCA(i)=MC(1,2);
end

[B,I] = sort(CCA)

for bst = 1:n_state
        
        rate(bst) = sum(cidx_Pha==I(bst))/(L-2*T_shift);
        ratea(bst) = sum(cidx_Pha(1:3465)==I(bst))/(L-2*T_shift);
        rateb(bst) = sum(cidx_Pha(3466:6919)==I(bst))/(L-2*T_shift);
end


% Initialize the matrix to store the results
times_matrix_k = zeros(16, 2); % Assuming there are 15 elements in DataCorrel
times_matrix_p = zeros(16, 2); % Assuming there are 15 elements in DataCorrel

% Loop through each element in DataCorrel
for i = 1:16
    % Get the size of the second dimension of LSD_TS
    size_value_k = size(DataCorrel(i).LSD_TS, 2);
    size_value_p = size(DataCorrel(i).PCB_TS, 2);

    % Store the size value in the first column
   times_matrix_k(i, 1) = size_value_k;
   times_matrix_p(i, 1) = size_value_p;
    % Calculate the cumulative sum and store it in the second column
    if i == 1
        times_matrix_k(i, 2) = size_value_k;
        times_matrix_p(i, 2) = size_value_p;
    else
        times_matrix_k(i, 2) = times_matrix_k(i-1, 2) + size_value_k;
        times_matrix_p(i, 2) = times_matrix_p(i-1, 2) + size_value_p;
    end
end




for s = 1:15
    for j= 1:n_state
        if s == 1
            scratekt(s,j)=sum(cidx_Pha(1:times_matrix_k(s, 2))==I(j))/(times_matrix_k(1, 2));
        else
            scratekt(s,j)=sum(cidx_Pha(times_matrix_k(s-1, 2) +1:times_matrix_k(s, 2))==I(j))/(times_matrix_k(1, 2));
        end
    end

end



for s = 1:15
    for j= 1:n_state
        if s == 1
            scratektp(s,j)=sum(cidx_Pha(3465 +1:3465 +times_matrix_p(s, 2))==I(j))/(times_matrix_p(1, 2));
        else
            scratektp(s,j)=sum(cidx_Pha(3465 +times_matrix_p(s-1, 2) +1:3465 +times_matrix_p(s, 2) -20)==I(j))/(times_matrix_p(1, 2));
        end
    end

end

grp = [zeros(15,1)',ones(15,1)'];
for i = 1:n_state
    c_1=scratekt(:,i);
    c_2=scratektp(:,i);
    CH(:,i) = [c_1' c_2'];
end


for i = 1:n_state
    subplot(3,n_state+1,i)
    imagesc(squareform(ctrs_Pha(I(i),:)),[-1 1])
    colormap(jet)
    axis square
    title(['State ' num2str(i) ' SFC' num2str(round(B(i),3))])
end

subplot(3,n_state+1,n_state+1)
imagesc(SC,[-1 1])
axis square
title('Connectome')


for k = 1:n_state
    subplot(3,n_state+1, n_state+1+k)
    boxplot(CH(:,k),grp)
end

subplot(3,n_state+1,2*n_state+2 +1)
bar(ratea/sum(ratea))
ylim([0 0.5])
ylabel('Probability LSD')

subplot(3,n_state+1,2*n_state+2 +2)
bar(rateb/sum(rateb))
ylim([0 0.5])
ylabel('Probability PLACEBO')


%%
%%


figure('Position', [100, 100, 1200, 1500]); % Adjust figure size as needed

% Loop through all 21 plots
for k = 1:16
    % Calculate the subplot position
    subplot(6, 3, k);
    
    % Create correlation matrix visualization
    h = imagesc(corrcoef(DataCorrel(k).LSD_TS'));
    
    % Set the colorbar limits to be fixed between -1 and 1
    caxis([-1 1]);
    
    % Add colorbar and set axis properties
    colorbar;
    axis square;     
    
    % Add title to each subplot
    title(['LSD\_FC ' num2str(k)]);
end

% Set a single colormap for the entire figure
colormap('jet'); % You can use other colormaps like 'parula', 'viridis', etc.

% Add a common title for the entire figure
sgtitle('Correlation Matrices from TS\_MDMA2 Data (Scale: -1 to 1)');

% Make the layout tighter
tight_spacing = 0.03;
set(gcf, 'Units', 'normalized');
set(gcf, 'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8]);

% Optional: Add a single, larger colorbar for the entire figure
% Delete individual colorbars first
delete(findobj(gcf, 'type', 'colorbar'));
h_bar = colorbar('Position', [0.93 0.1 0.02 0.8]);
ylabel(h_bar, 'Correlation Coefficient');

%%

figure('Position', [100, 100, 1200, 1500]); % Adjust figure size as needed

% Loop through all 21 plots
for k = 1:16
    % Calculate the subplot position
    subplot(6, 3, k);
    
    % Create correlation matrix visualization
    h = imagesc(corrcoef(DataCorrel(k).PCB_TS'));
    
    % Set the colorbar limits to be fixed between -1 and 1
    caxis([-1 1]);
    
    % Add colorbar and set axis properties
    colorbar;
    axis square;
    
    % Add title to each subplot
    title(['LSD\_PLA FC ' num2str(k)]);
end

% Set a single colormap for the entire figure
colormap('jet'); % You can use other colormaps like 'parula', 'viridis', etc.

% Add a common title for the entire figure
sgtitle('Correlation Matrices from TS\_MDMA2 Data (Scale: -1 to 1)');

% Make the layout tighter
tight_spacing = 0.03;
set(gcf, 'Units', 'normalized');
set(gcf, 'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8]);

% Optional: Add a single, larger colorbar for the entire figure
% Delete individual colorbars first
delete(findobj(gcf, 'type', 'colorbar'));
h_bar = colorbar('Position', [0.93 0.1 0.02 0.8]);
ylabel(h_bar, 'Correlation Coefficient');