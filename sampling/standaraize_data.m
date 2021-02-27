function norm_sample = standaraize_data(sample)
mean_v = mean(sample);
std_v = std(sample);
norm_sample = (sample - mean_v)./std_v;
