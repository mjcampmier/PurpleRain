classdef PurpleAirNetwork
    properties
        network
    end
    methods
        function obj = init(obj, fstr)
            load(fstr,'arr');
            s = arr{1,2};
            keys = fieldnames(s);
            keys2 = replace(keys,' ','_');
            clls = cell(length(keys),1);
            obj.network = cell2struct(clls, keys2);
            time = arr{2,2};
            time = time.Time;
            names = string(keys2);
            for k = 1:length(keys)
                name = names(k);
                pa = getfield(s, keys{k});
                lat = pa.Latitude;
                lon = pa.Longitude;
                pm25_cf_a = pa.A.PM_CF.PM25_CF; 
                pm25_cf_b = pa.B.PM_CF.PM25_CF;
                temp = pa.Meteorology.Temperature;
                rh = pa.Meteorology.RH;
                PA = PurpleAir(name, time, pm25_cf_a,...
                    pm25_cf_b, temp, rh, lat, lon);
                obj.network = setfield(obj.network, name, PA);
            end
        end
        function r = block_average(obj, dt)
            keys = fieldnames(obj.network);
            r = PurpleAirNetwork();
            clls = cell(length(keys),1);
            r.network = cell2struct(clls, keys);
            sensors = string(keys);
            for i = 1:length(sensors)
                sensor = string(sensors{i});
                PA = getfield(obj.network, sensor);
                PA_BA = PA.block_average(dt);
                r.network = setfield(r.network, sensor, PA_BA);
            end
        end
        function [dm, ds] = diurnal_average(obj,varargin)
            if isempty(varargin) || varargin{1} == "all"
                keys = fieldnames(obj.network);
                sensors = string(keys);
            else
                sensors = varargin{1};
            end
            dm = array2table(nan(24,length(sensors)));
            dm.Properties.VariableNames = cellstr(sensors);
            ds = array2table(nan(24,length(sensors)));
            ds.Properties.VariableNames = cellstr(sensors);
            for i = 1:length(sensors)
                sensor = sensors(i);
                PA = getfield(obj.network, sensor);
                [m,s] = PA.diurnal_average(false);
                dm(:,i) = table(m);
                ds(:,i) = table(s);
            end
            disp(dm);
            if length(varargin) == 2
                if varargin{2} == "all"
                    line(linspace(0,23,24),table2array(dm),...
                    'LineWidth',2);hold on; grid on;
                    xticks(linspace(0,23,24));
                    xlim([0,23]);ylim([0,...
                        nanmax(ceil(nanmax(table2array(dm))/10)*10)]);
                    xlabel('Hour of Day');ylabel('PM_{2.5} (\mug/m^{3})');
                else
                    idx = sensors==varargin{2};
                    dmt = table2array(dm(:,any(idx,2)));
                    line(linspace(0,23,24),dmt,...
                        'LineWidth',2);grid on;
                    hold on;legend(varargin{2},'Location','Southwest');
                    xticks(linspace(0,23,24));
                    xlim([0,23]);ylim([0,...
                        nanmax(ceil(nanmax(dmt)/10)*10)]);
                    xlabel('Hour of Day');ylabel('PM_{2.5} (\mug/m^{3})');
                end
            end
        end
        function [good, bad, down, pdf] = abperformance(obj, varargin)
            keys = fieldnames(obj.network);
            sensors = string(keys);
            good = nan(length(sensors),1);
            bad = nan(length(sensors),1);
            down = nan(length(sensors),1);
            pdf = nan(length(sensors),2);
            for i = 1:length(sensors)
                sensor = string(sensors(i));
                PA = getfield(obj.network, sensor);
                [g, b, d, p, ~] = PA.abperformance();
                good(i) = g;
                bad(i) = b;
                down(i) = d;
                pdf(i,:) = p;
            end
            if ~isempty(varargin) && varargin{1}==true
                x = categorical(sensors);
                br=bar(x,[good,down,bad],'stacked');br(1).FaceColor = [.2 .6 .5];
                set(gca,'xticklabel',sensors);
            end
        end
        function maptbl = map_network(obj)
            keys = fieldnames(obj.network);
            sensors = string(keys);
            lat = nan(length(sensors),1);
            lon = nan(length(sensors),1);
            x = nan(length(sensors),1);
            for i = 1:length(sensors)
                sensor = string(sensors(i));
                PA = getfield(obj.network, sensor);
                lat(i) = PA.lat;
                lon(i) = PA.lon;
                x(i) = nanmean([PA.pm25_cf_A, PA.pm25_cf_B]);
            end
            scatter(lon, lat, 200, x, 'filled');
            h = plot_google_map('MapScale',1,'ShowLabels',0);
            colormap('hot');c=colorbar;caxis([0, nanmax(x)]);
            yticks([]);xticks([]);set(gca,'Visible','off')
            ylabel(c, 'PM_{2.5} (\mug/m^{3})');
            maptbl = table(lon,lat,x);
            maptbl.Properties.VariableNames = {'Longitude','Latitude',...
                'PM25'};
            maptbl.Properties.RowNames = sensors;
        end
        function return_table = table(obj, varargin)
            keys = fieldnames(obj.network);
            r = PurpleAirNetwork();
            clls = cell(length(keys),1);
            r.network = cell2struct(clls, keys);
            sensors = string(keys);
            time = getfield(obj.network, sensors(1)).time;
            return_table = array2table(nan(length(time),length(sensors)));
            return_table.Properties.VariableNames = cellstr(sensors);
            for i = 1:length(sensors)
                sensor = string(sensors{i});
                PA = getfield(obj.network, sensor);
                return_table(:,i) = table(PA.pm25_cf_A');
            end
            tm = table(time');tm.Properties.VariableNames = {'Time'};
            return_table = [tm, return_table];
            if ~isempty(varargin) && varargin{1} == true
                return_table = table2timetable(return_table);
            end
        end
        function [r2, nrmse] = network_performance(obj, varargin)
            tbl = obj.table(true);
            r2 = nan(size(tbl,2), size(tbl,2));
            nrmse = nan(size(tbl,2), size(tbl,2));
            for i = 1:size(tbl,2)
                for j = 1:size(tbl,2)
                    if i == j
                        r2(i,j) = 1;
                        nrmse(i,j) = 0;
                    else
                        x =table2array(tbl(:,i)); y = table2array(tbl(:,j));
                        lm = fitlm(x, y);
                        r2(i,j) = lm.Rsquared.ordinary;
                        nrmse(i,j) = lm.RMSE/nanmean(y);
                    end
                end
            end
            if ~isempty(varargin) && varargin{1} == true
                figure(1);h1=heatmap(tbl.Properties.VariableNames,...
                    tbl.Properties.VariableNames, r2);
                figure(2);h2=heatmap(tbl.Properties.VariableNames,...
                    tbl.Properties.VariableNames, nrmse);
            end
        end
        function flags = qf(obj)
            keys = fieldnames(obj.network);
            sensors = string(keys);
            time = getfield(obj.network, sensors(1)).time;
            flags = nan(length(time), length(sensors));
            for i = 1:length(sensors)
                sensor = string(sensors(i));
                PA = getfield(obj.network, sensor);
                flags(:,i) = PA.qf();
            end
            flags = array2table(flags);
            flags.Properties.VariableNames = sensors;
        end
    end
end