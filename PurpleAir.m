classdef PurpleAir
    properties
        name
        time
        pm25_cf_A
        pm25_cf_B
        temperature
        relative_humidity
        lat
        lon
    end
    methods
        function obj = PurpleAir(name, time, pm25_cf_A, pm25_cf_B,...
                temperature, relative_humidity, lat, lon)
            obj.name = replace(name,'_',{' '});
            if isa(time, 'double')== 1
                obj.time = datetime(time, 'ConvertFrom', 'juliandate');
            else
                obj.time = time;
            end
            obj.pm25_cf_A = pm25_cf_A;
            obj.pm25_cf_B = pm25_cf_B;
            obj.temperature = temperature;
            obj.relative_humidity = relative_humidity;
            obj.lat = lat;
            obj.lon = lon;
        end
        function r = block_average(obj, dt)
            TT = timetable(obj.time', obj.pm25_cf_A', obj.pm25_cf_B',...
                obj.temperature', obj.relative_humidity');
            TT = retime(TT,'regular',@nanmean,'TimeStep', dt);
            TT.Properties.VariableNames = {'pm25_cf_A',...
                'pm25_cf_B','temperature','relative_humidity'};
            r = PurpleAir(obj.name, TT.Time, TT.pm25_cf_A, TT.pm25_cf_B,...
                TT.temperature, TT.relative_humidity, obj.lat, obj.lon);
        end
        function timeseries_plot(obj)
            stairs(obj.time, obj.pm25_cf_A); hold on;
            stairs(obj.time, obj.pm25_cf_B);
            ylim([0, max(obj.pm25_cf_A)]);
        end
        function [dm, dr] = diurnal_average(obj, varargin)
            TT = timetable(obj.time', obj.pm25_cf_A', obj.pm25_cf_B',...
                obj.temperature', obj.relative_humidity');
            TT.Properties.VariableNames = {'pm25_cf_A',...
                'pm25_cf_B','temperature','relative_humidity'};
            H = findgroups(hour(TT.Time));
            dm = splitapply(@nanmedian,TT.pm25_cf_A, H);
            dr = splitapply(@iqr, TT.pm25_cf_A, H);
            if length(varargin)>=1
                if varargin{1} == true
                    line(linspace(0,23,24),dm,'Color',[0.5,0,0.5],'LineWidth',3);hold on;
                    fill([linspace(0,23,24), fliplr(linspace(0,23,24))],...
                        [(dm-dr)', fliplr((dm+dr)')],[0.5,0,0.5],'EdgeColor','none',...
                        'FaceAlpha',0.5);
                    xticks(linspace(0,23,24));
                    xlim([0,23]);
                    ylim([0,ceil(max(dm+dr)/10)*10]);
                    xlabel('Hour of Day');ylabel('PM_{2.5} (\mug/m^{3})');
                    title(obj.name);legend('Median','IQR','Location','Southwest');
                    grid on;
                end
            end
        end
        function [good, bad, down, pdf, lm] = abperformance(obj, varargin)
            error = obj.pm25_cf_B-obj.pm25_cf_A;
            pm25 = nanmean([obj.pm25_cf_A;obj.pm25_cf_B]);
            down = sum(isnan(pm25))/length(pm25); 
            bad =  sum(abs(error)>5)/length(pm25);
            good = 1-(down+bad);
            pdf = [nanmean(error), nanstd(error)];
            if length(varargin) >= 1
                lm = fitlm(obj.pm25_cf_A,obj.pm25_cf_B);
                res = lm.Residuals.Raw;
                slope = lm.Coefficients.Estimate(2);
                inter = lm.Coefficients.Estimate(1);
                mx = max([ceil(max(obj.pm25_cf_A)/10)*10,...
                ceil(max(obj.pm25_cf_B)/10)*10]);
                subplot(1,3,1);
                sgtitle(obj.name);
                histogram(error,'Normalization',...
                    'pdf','FaceColor','red','FaceAlpha',1);hold on; grid on;
                xlim([-5,5]);
                title('Raw Residuals Histogram')
                subplot(1,3,2);
                scatter(obj.pm25_cf_A,obj.pm25_cf_B,100,'blue','filled');
                line(linspace(0,mx,100),...
                    slope*linspace(0,mx,100)+inter,'Color','blue',...
                    'LineWidth',2);
                hold on; grid on; 
                legend('2-min',strcat('B = ',string(round(slope,3)),...
                    {'*A + '}, string(round(inter, 3))),'Location','Northwest');
                xlim([0,mx]);ylim([0,mx]);
                title('A-B Agreement');
                subplot(1,3,3);
                histogram(res,'Normalization',...
                    'pdf','FaceColor','blue','FaceAlpha',1);hold on; grid on;
                xlim([-5,5]);
                title('Model Residuals Histogram');
            else
                lm = nan;
            end
        end
        function flag = qf(obj)
            error = obj.pm25_cf_B-obj.pm25_cf_A;
            flag = zeros(1, length(error));
            flag(error>5 | error<-5) = 1;
        end
    end
end