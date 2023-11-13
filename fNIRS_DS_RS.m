function [data_ds_rs] = fNIRS_DS_RS(data)
fs=25;
[sample, chan, trial]=size(data);

for tr=1:trial

    for ch = 1:chan
        kk=squeeze(data(1:end-1,ch,tr));
        kk = reshape(kk,[length(kk)/fs fs]);
        data_ds_rs(:,ch,tr)=mean(kk,2);
    end
end

[sample, chan, trial]=size(data_ds_rs);

data_ds_rs = reshape(data_ds_rs,[trial,chan,sample]);


