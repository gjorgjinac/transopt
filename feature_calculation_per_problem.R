# Run this command in Base environment
# R -e 'install.packages("flacco", dependencies = TRUE, repos = "http://cran.us.r-project.org", Ncpus = -1)'

library(flacco)

calculate_features <- function(X, Y) {

  dim = ncol(X)
  samples = nrow(X)
  samples_per_cell= 10
  #block = round((samples/10)^(1/dim))

  feat.object <- createFeatureObject(X = X, y = Y)

  features = c()
    
  ela_types <- c("cm_angle", "cm_conv", "cm_grad", "ela_distr", "ela_level", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "bt", "gcm", "ic")
    
  # Needs sampling
  # "ela_conv", "ela_curv", "ela_local", "ela_conv"

  for (ela_type in ela_types) {

      tryCatch(
        expr = {
          fe = calculateFeatureSet(feat.object, ela_type)
          features <- c(features,fe)
        },
        error = function(w){
            print(paste("Can not compute", ela_type, sep=" "))
        }
      )
      
  }

  return(features)
}



calculate_ELA<-function(X,y){
	feat.object = createFeatureObject(X = X,y=y)
	ctrl = list(allow_cellmapping = FALSE,blacklist=c("pca","ela_distr", "ic","nbc"))
	features = calculateFeatures(feat.object, control = ctrl)
	
	features_pca <- NULL
	features_pca$pca.expl_var.cov_x <- NA;
	features_pca$pca.expl_var.cor_x <- NA;
	features_pca$pca.expl_var.cov_init <- NA;
	features_pca$pca.expl_var.cor_init <- NA;
	features_pca$pca.expl_var_PC1.cov_x <- NA;
	features_pca$pca.expl_var_PC1.cor_x <- NA;
	features_pca$pca.expl_var_PC1.cov_init <- NA;
	features_pca$pca.expl_var_PC1.cor_init <- NA;
	features_pca$pca.costs_fun_evals <- NA;
	features_pca$pca.costs_runtime <- NA;

	tryCatch({features_pca <-calculateFeatureSet(feat.object, set = "pca")},
		error = function(e){print("Error in PCA")}
	)

	
	features_ela_distr <- NULL
	features_ela_distr$ela_distr.skewness <- NA;
	features_ela_distr$ela_distr.kurtosis <- NA;
	features_ela_distr$ela_distr.number_of_peaks <- NA;
	features_ela_distr$ela_distr.costs_fun_evals <- NA;
	features_ela_distr$ela_distr.costs_runtime <- NA;

	tryCatch({features_ela_distr <-calculateFeatureSet(feat.object, set = "ela_distr")},
		error = function(x){print("Error in ela_distr");}
	)
	
	features_ic <- NULL
	features_ic$ic.h.max <- NA;
	features_ic$ic.eps.s <- NA;
	features_ic$ic.eps.ratio <- NA;
	features_ic$ic.m0 <- NA;
	features_ic$ic.costs_fun_evals <- NA;
	features_ic$ic.costs_runtime <- NA;
	features_ic$ic.eps.max <- NA;
	
	tryCatch(features_ic <-calculateFeatureSet(feat.object, set = "ic"),
		error = function(x){print("Error in ic");}
	)
			
	features_nbc <- NULL
	
	features_nbc$nbc.nn_nb.sd_ratio <- NA;
	
	features_nbc$nbc.nn_nb.mean_ratio <- NA;
	features_nbc$nbc.nn_nb.cor <- NA;
	features_nbc$nbc.dist_ratio.coeff_var <- NA;
	features_nbc$nbc.nb_fitness.cor <- NA;
	features_nbc$nbc.costs_fun_evals <- NA;
	features_nbc$nbc.costs_runtime <- NA;
	

	tryCatch({features_nbc <-calculateFeatureSet(feat.object, set = "nbc")},
		error = function(e){print("Error in NBC")}
	)
	


	features <- append(features, features_pca)
	features <- append(features, features_ela_distr)
	features <- append(features, features_ic)
	features <- append(features, features_nbc)
	
		
	temp_obj<-unlist(features, use.names=TRUE)
	return(temp_obj)
	
	
}


args = commandArgs(trailingOnly=TRUE)
file = args[1]
problem_id = as.integer(args[2])
df=read.csv(paste("data/", file, sep=""));
nrow(df)

datalist = list()

numclasses = 24
numinstances =999

i=1
write_file=paste("ela_data/", file, "_problem_id_", problem_id, ".csv",sep="");

if (file.exists(write_file))
{
    print("File already exists. Skipping computing ELA features");
} else {
    for (class in list(problem_id) ) {

      for (instance in 1:numinstances) {

        sdf = df[df[, 2]==instance & df[, 1]==class, ];
        
        X = sdf[, 5:ncol(sdf)-1];
        y = sdf[, ncol(sdf)];
        if (class==1 & instance==1) {
            print(X)
            print(y)
        }
        print_info = paste(class, instance, nrow(X), ncol(X), nrow(y), ncol(y), sep=" ");
        print(print_info)

        features = calculate_ELA(X, y);

        features <- c(features, instance_id=instance);
        features <- c(features, problem_id=class);

        #datalist <- append(datalist, features);

        datalist[[i]] <- features;
          
        i<-i+1
      }

      big_data = do.call(rbind, datalist);
      #print(big_data);
    }


    write.csv(big_data, write_file, row.names = FALSE);
}



