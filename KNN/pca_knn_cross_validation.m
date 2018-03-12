input_data = load('knn_data');
train_data_original=getfield(input_data,'train_data');
% reduce dimensions of the train_data_org

mean_vector_for_features = sum(train_data_original)/size(train_data_original,1);
normalized_train_data = train_data_original - mean_vector_for_features;
covariance = (transpose(normalized_train_data)* normalized_train_data)/(size(normalized_train_data,1)-1);
[e_vectors,e_values] = eigs(covariance, 166);
% Take top 50 columns
eigen_vectors_50 = e_vectors(:, 1:50);

train_data_reduced = normalized_train_data * eigen_vectors_50;
train_labels_original=getfield(input_data,'train_label');
train_datalabels=horzcat(train_data_reduced,train_labels_original);
train_data_labels_CV = train_datalabels(randperm(size(train_datalabels,1)),:);
accuracies_for_each_fold = zeros(5,9);
k_values = [1 3 5 7 9 11 13 15 17];

accuracy_test_set = zeros(1,9);
test_data_original = getfield(input_data,'test_data');
test_labels_original = getfield(input_data,'test_label');

bucket_size = 5000/5;
for f=1:5
    train = [];
    test = [];
    new_train_labels = [];
    new_test_labels = [];
    for r=1:5000
        if(ceil(r/bucket_size) == f)
            test = vertcat(test,train_data_labels_CV(r, 1:50));
            new_test_labels = vertcat(new_test_labels,train_data_labels_CV(r,51));
        else
            train = vertcat(train,train_data_labels_CV(r, 1:50));
            new_train_labels = vertcat(new_train_labels,train_data_labels_CV(r,51));
        end
    end
    accuracies_for_each_fold(f,:) = kNN(train, test, new_train_labels, new_test_labels);
end

updated_accuracies = sum(accuracies_for_each_fold)*100/5;
plot(k_values,updated_accuracies);
kVsAccuracies = horzcat(transpose(k_values),transpose(updated_accuracies));
[max_training_accuracy, k_value_index ] = max(updated_accuracies);
assigned_k_value = k_values(k_value_index);

save("accuracy2.mat","kVsAccuracies")
title('Neighbors Vs Accuracy when dimensions are reduced using PCA')
xlabel('K')
ylabel('Accuracy for each K after PCA')

mean_vector_for_test = sum(test_data_original)/size(test_data_original,1);
normalized_test_data = test_data_original - mean_vector_for_test;
covariance_test = (transpose(normalized_test_data)* normalized_test_data)/(size(normalized_test_data,1)-1);
[e_vectors_test,e_values_test] = eigs(covariance_test, 166);
% Take top 50 columns
eigen_vectors_test_50 = e_vectors_test(:, 1:50);

reduced_test = normalized_test_data * eigen_vectors_test_50;

accuracy_test_set(1,:) = kNN(train_data_reduced, reduced_test, train_labels_original, test_labels_original);
accuracy_for_max_k = accuracy_test_set(k_value_index);
