%Loading the Dataset.
data = load("kmeans_data.mat");
%Converting from structure to cell structure matlab.
cell = struct2cell(data);

Input_data = cell2mat(cell);

number_clusters = [1 2 3 4 5 6 7 8 9 10];
k = 2;


KvsCostFunc = zeros(size(number_clusters,2),2);
for num_clusters =1 : size(number_clusters,2)
    k = number_clusters(num_clusters);
    KvsCostFunc(num_clusters,1) = k;
    % Get k random numbers
    Indexes_of_Clusters = randperm( 7195, k); 
    
    centers = [];
    for i=1: k
        centers(i,:) = Input_data(Indexes_of_Clusters(i),:);
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