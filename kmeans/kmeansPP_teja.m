%Loading the Dataset.
data = load("kmeans_data.mat");
%Converting from structure to cell structure matlab.
cell = struct2cell(data);
Input_data = cell2mat(cell);

number_clusters = [1 2 3 4 5 6 7 8 9 10];

KvsCostFunc = zeros(size(number_clusters,2),2);
for num_clusters =1 : size(number_clusters,2)
    cluster_intialised = 0;
    k = number_clusters(num_clusters);
    KvsCostFunc(num_clusters,1) = k;
    
   
    
    % Get Random number first
    first_cluster_center = randperm(7195, 1);
    
    %variable for collecting clusternums
    cluster_intialised = cluster_intialised+1;
    clusterCentersNums(cluster_intialised) = first_cluster_center;
    
    %cluster centers 
    cluster_coordinates(cluster_intialised,:) = Input_data(clusterCentersNums(cluster_intialised),:);
    while(cluster_intialised<=k)
        distance_array_from_points = zeros(7,cluster_intialised);
        for i = 1:7195
            for cluster=1: cluster_intialised
                distance = 0;
                for d=1:21
                    distance = distance + (Input_data(i,d) - cluster_coordinates(cluster,d))^2;
                end
                distance_array_from_points(i,cluster) = distance;    
            end
        end
        compute_min_distance = min(distance_array_from_points, [], 2);
        [dis,index_of_maxdist] = max(compute_min_distance);
        %[dist_in_max, index_of_dist_max] = max(dist);
        %res_index = index_of_max_dist_arr(index_of_dist_max);
        cluster_intialised = cluster_intialised+1;
        clusterCentersNums(cluster_intialised) = index_of_maxdist;
        cluster_coordinates(cluster_intialised,:) = Input_data(index_of_maxdist,:);
    end
    
    Centers_update = zeros(k,21); 
    while(isequal(centers,Centers_update) == 0)
        nearest_center_for_point = zeros(7195,2);
        for point =1: 7195
            nearest_cluster_center= 0;
            smallest_possible_dist = intmax('int64');
            for cluster=1: k
                distance = 0;
                for feature =1:21
                    distance= distance + (Input_data(point,feature) - centers(cluster,feature))^2;
                end
                distance = sqrt(distance);

                if(distance <= smallest_possible_dist)
                    nearest_cluster_center = cluster;
                    smallest_possible_dist = distance;
                end
            end
            nearest_center_for_point(point,1) = point;
            nearest_center_for_point(point,2) = nearest_cluster_center;
        end

        % Get column on which selection can be performed
        column_clusters = nearest_center_for_point(:, end);
        Centers_update = centers;
        objective_Function = 0;
        for i=1:k
            Points_in_Cluster = nearest_center_for_point(column_clusters==i);
            Updated_Clusters = zeros(size(Points_in_Cluster,1),21); % 2 here is dimension;
            for index= 1: size(Points_in_Cluster,1)
                Updated_Clusters(index,:) = Input_data(Points_in_Cluster(index),:);
            end
            centers(i,:) = sum(Updated_Clusters) ./ size(Points_in_Cluster,1) ;
            for point=1:size(Updated_Clusters,1)
                for d=1:size(Updated_Clusters,2)
                    objective_Function = objective_Function + (Updated_Clusters(point,d) - centers(i,d))^2;
                end
            end
        end    
    end
    
    KvsCostFunc(num_clusters,2) = objective_Function;
end
plot(KvsCostFunc);