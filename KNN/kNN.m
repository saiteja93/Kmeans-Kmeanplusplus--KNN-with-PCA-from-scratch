function[accuracies_fold]= kNN(train_data, test_data, train_labels, test_labels)
    
    k_list = [1 3 5 7 9 11 13 15 17];
    accuracies_fold = zeros(size(k_list,2),1);

    train_length = length(train_data);
    test_length = length(test_data);
    
    number_of_features = size(train_data,2);
    
    distances_between_neighbors = zeros(train_length,2);
    max_distances_between_neighbors = zeros(train_length,2);

    max_number_neighbors_considered = max(k_list);

    max_nearest_neighbors_considered = zeros(test_length, max_number_neighbors_considered);
    for i = 1:test_length
        for j = 1:train_length
            dist = 0;
            % For all features
            for k=1:number_of_features
                dist = dist + (test_data(i,k)-train_data(j,k))^2;
            end

            dist = sqrt(dist);

            distances_between_neighbors(j,1) = train_labels(j,:);
            distances_between_neighbors(j,2) = dist;

        end
        max_distances_between_neighbors = sortrows(distances_between_neighbors,2);
        max_nearest_neighbors_considered(i,:) = max_distances_between_neighbors(1:max_number_neighbors_considered,1);
    end

    predictions = zeros(test_length, length(k_list));
    for i=1:test_length
        for k =1:length(k_list)
            predictions(i,k) = mode(max_nearest_neighbors_considered(i,1:k_list(k)));
        end
    end


    for k =1:length(k_list)
        true_total = 0;
        for i=1:test_length
            if predictions(i,k) == test_labels(i)
                true_total = true_total+1;
            end
        end
        accuracies_fold(k) = true_total/test_length;
        
    end

end
