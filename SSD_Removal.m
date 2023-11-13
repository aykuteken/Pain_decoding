function [data]= SSD_Removal(data,subs)

for i=1:length(subs)

    long = squeeze(data(:,1,subs(i,1),:));
    short = squeeze(data(:,1,subs(i,2),:));

    for j=1:size(long,2) %% per trial

        beta(j)=pinv(short(:,j)'*short(:,j))*short(:,j)'*long(:,j);
        
        data(:,1,subs(i,1),j)=squeeze(data(:,1,subs(i,1),j))-beta(j)*squeeze(data(:,1,subs(i,2),j));
    end

end

short_chan=unique(subs(:,2));
data=squeeze(data);

data(:,short_chan,:)=[];

end
