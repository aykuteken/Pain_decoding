function [data] = Trial_Elimination(data)
th=5*10^-4;
for i=1:size(data,3) %% number of trials

    for j=1:size(data,2) %% channels

        data(:,j,i) = detrend(squeeze(data(:,j,i)));

        if any(data(:,j,i)>th) || any(data(:,j,i)<-1*th)    %% Threshold

            data(:,j,i)=0;

        end
    end

end
