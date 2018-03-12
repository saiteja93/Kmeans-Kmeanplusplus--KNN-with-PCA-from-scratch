input_data = load('knn_data');
train_data_orginal = getfield(input_data,'train_data');
train_labels_orginal = getfield(input_data,'train_label');
train_datalabels_=horzcat(train_data_orginal,train_labels_orginal);
shuffled_data_CV = train_datalabels_(randperm(size(train_datalabels_,1)),:);
accuracy_for_each_folds = zeros(5,9);
k_values = [1 3 5 7 9 11 13 15 17];

accuracy_test = zeros(1,9);
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
            test = vertcat(test,shuffled_data_CV(r, 1:166));
            new_test_labels = vertcat(new_test_labels,shuffled_data_CV(r,167));
        else
            train = vertcat(train,shuffled_data_CV(r, 1:166));
            new_train_labels = vertcat(new_train_labels,shuffled_data_CV(r,167));
        end
    end
    accuracy_for_each_folds(f,:) = kNN(train, test, new_train_labels, new_test_labels);
end

updated_accuracies = sum(accuracy_for_each_folds)*100/5;
plot(k_values,updated_accuracies);
KvsAccuracies = horzcat(transpose(k_values),transpose(updated_accuracies));
[max_training_accuracy, k_value_index ] = max(updated_accuracies);
assigned_k_value = k_values(k_value_index);

save("accuracy2.mat","KvsAccuracies")
title('Neighbors Vs Accuracy for K-fold Cross validation')
xlabel('K')
ylabel('Accuracy for each k')

accuracy_test(1,:) = kNN(train_data_orginal, test_data_original, train_labels_orginal, test_labels_original);
accuracy_for_max_k = accuracy_test(k_value_index);
