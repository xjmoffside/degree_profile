function [w,coupling] = dwass_discrete2(x,y,px,py)
% compute W_1 between discrete distributions \sum px(i)\delta_{x(i)} and \sum px(i)\delta_{x(i)}
% where the cardinality of the support may not be the same
% input: x,y,px,py are column vectors (already sorted in ascending order)
if length(x)==length(px) && length(y)==length(py) 
    
   nx=length(x);ny=length(y);
  
   % sort x 
  % [x_sort,I_x]=sort(x);
  % [y_sort,I_y]=sort(y);
   
  
%    ind_x=nx;
%    ind_y=ny;
%    mass_x=px(I_x(ind_x));
%    mass_y=py(I_y(ind_y));
%    w=0;
%    coupling=zeros(nx,ny);
%    while ind_x>0
%        % move mass temp from x to y
%        temp=min(mass_x,mass_y);
%        coupling(I_x(ind_x),I_y(ind_y))=coupling(I_x(ind_x),I_y(ind_y))+temp;
%        w=w+abs(x_sort(ind_x)-y_sort(ind_y))*temp;
%        mass_x=mass_x-temp;
%        mass_y=mass_y-temp; 
%        if mass_x==0
%            ind_x=ind_x-1;
%            if ind_x>=1
%                mass_x=px(I_x(ind_x));
%            end
%        end     
%        if mass_y==0
%            ind_y=ind_y-1;
%            if ind_y>=1
%                mass_y=py(I_y(ind_y));
%            end
%        end
%    end
%    
%    if ind_y>0 || mass_x>0 || mass_y>0
%        error('coupling error')
%    end
   
  
    ind_x=nx;
    ind_y=ny;
    mass_x=px(ind_x);
    mass_y=py(ind_y);
    w=0;
    coupling=zeros(nx,ny);
    while ind_x>0 && ind_y>0
        % move mass temp from x to y
%         if mass_x<mass_y
%             temp=mass_x;
%             indx=indx-1;
%         end
%         if mass_x>mass_y
%             temp=mass_y;
%             indy=indy-1;
%         end
%         if mass_x==mass_y
%             temp=mass_x;
%             indx=indx-1;
%             indy=indy-1;
%         end
        temp=min(mass_x,mass_y);
        coupling(ind_x,ind_y)=coupling(ind_x,ind_y)+temp;
        w=w+abs(x(ind_x)-y(ind_y))*temp;
        mass_x=mass_x-temp;
        mass_y=mass_y-temp; 
        if mass_x==0
            ind_x=ind_x-1;
            if ind_x>=1
                mass_x=px(ind_x);
            end
        end     
        if mass_y==0
            ind_y=ind_y-1;
            if ind_y>=1
               mass_y=py(ind_y);
            end
        end
    end
    
 %   if ind_y>0 || mass_x>0 || mass_y>0
 %       error('coupling error')
 %   end



else
    error('length not equal')
end
    