##############################
##############################
##############################
##############################

# libraries
library(tidyverse)
library(visdat)
library(skimr)
library(pROC)
library(caret)
library(gbm)
library(C50)
library(ranger)

#################################
########### DATA IMPORT ######### 
#################################

# import the full train set
lc_full_train <- read.csv("lending_club_train.csv", stringsAsFactors = FALSE)

# import the unlabelled test set (don't touch!)
lc_test <- read.csv("lending_club_test.csv", stringsAsFactors = FALSE)


# create a Combined Dataset for feature engineering (i.e. select same vars/ impute data: nb: works with simple imputation performed below, but not if to switch to multiple imputation due to data leakage)

# use union() to create a "long" combination of training and test data
lc_features <- union(
  # the mutate function adds a source so we can split it again later
  lc_full_train %>% mutate(data_source = "train"),
  lc_test %>% mutate(data_source = "test")
)

# confirm that the union worked correctly
table(lc_features$data_source)


#################################
####### DATA EXPLORATION ######## 
#################################

# quick overview of the dataset
str(lc_full_train)

# many of the features refer to joint loans, but this is uncommon (these vars can be removed, they also have no values: sec_)
lc_full_train %>% count(application_type)

# data types and missingness
# many of features have a lot of missing data, many are referring to secondary applicants
# will need to check closer if the character variables are character or factors
pdf("train_original_data_vis.pdf")
vis_dat(lc_full_train[1:7500,], sort_type = FALSE) 
dev.off()

vis_dat(lc_full_train[250000:257500,], sort_type = FALSE) 
vis_dat(lc_full_train[480000:487500,], sort_type = FALSE) 


#################################
####### FEATURE DOWNSIZE ######## 
#################################

lc_features <- lc_features %>% 
  select(-starts_with("sec_"), # removes all vars that start with "sec" (no values)
         -revol_bal_joint)  # no values

# look at data types and missingness
lc_features %>% 
  filter(data_source == "train"[1:5000]) %>% 
  vis_miss()

lc_features %>% 
  filter(data_source == "train"[1:5000]) %>% 
  vis_dat(sort_type = FALSE)  

# impute numeric data
lc_features_imp <- lc_features %>%
  mutate_if(anyNA, list(mvi = ~if_else(is.na(.), 1, 0))) %>%  # create dummies for missing
  select(-emp_title_mvi, # these are categorical/ characterand will be handled separately
         -title_mvi,
         -verification_status_joint_mvi,
         -default_mvi, # target should not have one (obviously omitted in test)
         -dti_mvi) %>% # only one example with missing data
  mutate(default = factor(default, # also create factor from default(required for some models)
            levels = c(0, 1), 
            labels = c("no","yes"))) %>% # factorise default (handy for some models + does not impute on next line)
  mutate_if(is.numeric , replace_na, replace = -1) %>% # replace numeric NAs with -1
  mutate(verification_status_joint = recode(verification_status_joint, 
                                            "Not Verified" =  "Not Verified",
                                            .default = "Other")) %>% 
  replace_na(list(verification_status_joint = "Missing")) # keep spaces and NA as separate categories(large, but not sure what the difference is)

# check if all values imputed now  
lc_features_imp %>% 
  filter(data_source == "train"[1:5000]) %>% 
  vis_miss()

## now fix the character variables (only the categorical for now )
lc_features_imp %>% 
  select_if(is.character) %>% 
  View()

# variation in:
lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(term) # factor

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(emp_length) # factor (leave as is for now)

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(home_ownership) # factor (collapsed: any and other)

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(purpose) # factor, collapse later

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(addr_state) # factor, collapsed very small states (semi-arbitrary by eye-ball)

lc_features_imp %>% filter(data_source == "train") %>% 
ggplot() +
  aes(x = addr_state, color = default) +
  geom_bar()

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(initial_list_status) # factor

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(application_type) # factor

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(verification_status_joint) # factor


# for later
lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(pymnt_plan) # all n (i.e. drop now)

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(disbursement_method) # all "Cash", drop now)

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(desc) # much variation, drop now

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(emp_title)  # engineered below

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(title) # drop, too much variation and encoded in other feature (purpose)

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(zip_code) # not relevant

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(earliest_cr_line) # engineered below

lc_features_imp %>% 
  filter(data_source == "train") %>% 
  count(revol_util) # take away percentage sign, convert to numeric, see below

## create factor variables
lc_features_imp <- lc_features_imp %>% 
  mutate(term = factor(term),
         emp_length = factor(emp_length),
         home_ownership = factor(home_ownership),
         home_ownership = fct_recode(home_ownership, OTHER = "ANY"),
         purpose = factor(purpose),
         addr_state = factor(addr_state),
         addr_state = fct_recode(addr_state, small_states = "AK",
                                 small_states = "DC", small_states = "DE",
                                 small_states = "IA", small_states = "ID",
                                 small_states = "ME", small_states = "ND",
                                 small_states = "NE", small_states = "SD",
                                 small_states = "VT", small_states = "WY"),
         initial_list_status = factor(initial_list_status),
         application_type = factor(application_type),
         verification_status_joint = factor(verification_status_joint),
         revol_util = as.numeric(gsub(pattern = "\\%", replacement = "", revol_util)), # take out % and convert to numeric
         revol_util_mvi = ifelse(is.na(revol_util), 1, 0), # create variable for missing
         revol_util = replace_na(revol_util, replace = -1),
         earliest_cr_line = as.numeric(gsub(pattern = ".*-", replacement = "", earliest_cr_line)), # only keep year, as numeric (likely says something about the age of applicant!)
         earliest_cr_cat = factor(case_when(earliest_cr_line < 1988 ~ "Before 1988",
                                     earliest_cr_line < 2000 ~ "1988 - 1999",
                                     TRUE ~ "2000 -")), # create bins 10%, 45%, 45%
         career_level = tolower(emp_title),
         career_level =  factor(case_when(
             str_detect(career_level, "manager") ~ "Manager",           
             str_detect(career_level, "lead") ~ "Manager",
             str_detect(career_level, "coordinator") ~ "Manager",
           str_detect(career_level, "director") ~ "Executive",
           str_detect(career_level, "assistant") ~ "Administrative",
           str_detect(career_level, "administrator") ~ "Administrative",
           str_detect(career_level, "analyst") ~ "Analyst",
           str_detect(career_level, "sales") ~ "Sales",
           str_detect(career_level, "engineer") ~ "Engineer",
           str_detect(career_level, "inc") ~ "inc",
           str_detect(career_level, "officer") ~ "Officer",
           str_detect(career_level, "supervisor") ~ "Supervisor",
           str_detect(career_level, "driver") ~ "Driver",
           str_detect(career_level, "nurse") ~ "Nurse",
           str_detect(career_level, "specialist") ~ "Specialist",
           str_detect(career_level, "senior") ~ "Senior",
           str_detect(career_level, "teacher") ~ "Teacher",
           str_detect(career_level, "service") ~ "Service",
           str_detect(career_level, "services") ~ "Service",
           str_detect(career_level, "technician") ~ "Technician",
           str_detect(career_level, "tech") ~ "Tech",
           str_detect(career_level, "executive") ~ "Executive",
           str_detect(career_level, "president") ~ "Executive",
           TRUE ~ "Other" # if this statement is TRUE, then give it other
           ))

         )

# check the earliest_cr_line to create bins (skewed), fixed above
quantile(lc_features_imp$earliest_cr_line, c(0.01, .1, .55, 0.99))

lc_features_imp %>% 
  filter(data_source == "train"[1:5000]) %>% 
  vis_dat()

## for now drop the remaining character variables
lc_features_imp <- lc_features_imp %>% 
  select(-pymnt_plan, # (no variation)
         -disbursement_method, # (no variation)
         -desc,# (too random)
         -emp_title, # roughly taken care of
         -title, # encoded already into another variable (purpose)
         -zip_code, # not relevant
         -earliest_cr_line # handled above (note: could be an indication of applicant age)
         )

######################################################
####### DIVIDE INTO TRAIN, VALIDATION AND TEST ######## 
######################################################

# fully imputed train set
lc_train_imp <- lc_features_imp %>% 
  filter(data_source == "train") %>% 
  select(-data_source)

# divide full train into smaller train and validation sets
set.seed(1705) # set seed for replicability
# smaller train
tr_lc <- sample_n(lc_train_imp, 325000) # randomly sample 50% of total data
lc_train_imp_small <- lc_train_imp %>%  
  semi_join(tr_lc, by = "id") 
# validation
lc_val_imp_small <- lc_train_imp %>% 
  anti_join(tr_lc, by = "id") # save those not in train to validation set

# fully imputed test set
lc_test_imp <- lc_features_imp %>% 
  filter(data_source == "test") %>% 
  select(-data_source)

# check balance
prop.table(table(lc_train_imp$default))
prop.table(table(lc_train_imp_small$default))
prop.table(table(lc_val_imp_small$default))


#####
#### LOGIT
####

# run logit for a quick test of the model: loan defaults is binary.
logit_bad_2 <- glm(default ~ . -id, 
                   data = lc_train_imp_small, family = "binomial") # run model with all vars but id

summary(logit_bad_2)

# run model with all vars but id and those whose rank is not id-ed above (not ideal, but fast)
logit_bad_3 <- glm(default ~ . -id -tot_coll_amt_mvi -tot_cur_bal_mvi -chargeoff_within_12_mths_mvi
                                           -mo_sin_old_rev_tl_op_mvi -mo_sin_rcnt_rev_tl_op_mvi
                                           -mo_sin_rcnt_tl_mvi -mort_acc_mvi -num_accts_ever_120_pd_mvi
                                           -num_actv_bc_tl_mvi -num_actv_rev_tl_mvi -num_bc_tl_mvi
                                           -num_il_tl_mvi -num_op_rev_tl_mvi -num_rev_tl_bal_gt_0_mvi
                                           -num_sats_mvi -num_tl_30dpd_mvi -num_tl_90g_dpd_24m_mvi -num_tl_op_past_12m_mvi
                                           -tot_hi_cred_lim_mvi -total_bal_ex_mort_mvi -total_bc_limit_mvi -total_il_high_credit_limit_mvi
                                           -annual_inc_joint_mvi -verification_status_joint -dti_joint_mvi -open_acc_6m_mvi -open_act_il_mvi 
                                           -open_il_12m_mvi -open_il_24m_mvi -total_bal_il_mvi -open_rv_12m_mvi -open_rv_24m_mvi
                                           -max_bal_bc_mvi -all_util_mvi -inq_fi_mvi -total_cu_tl_mvi -inq_last_12m_mvi -revol_util_mvi,
                                            data = lc_train_imp_small, family = "binomial") # run garbage model with all vars but id (and all NA)

summary(logit_bad_3)


preds_val <- predict(logit_bad_3, lc_val_imp_small, type = "response")
preds_class <- ifelse(preds_val > 0.5, "pos", "neg")

lc_val_imp_small$preds_logit_bad <- preds_val 
lc_val_imp_small$preds_class <- preds_class

prop.table(table(lc_val_imp_small$preds_class, lc_val_imp_small$default))

logit_bad_3_roc <- roc(predictor = lc_val_imp_small$preds_logit_bad, response = lc_val_imp_small$default)
plot(logit_bad_3_roc, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE)
auc(logit_bad_3_roc) # substantially better than day 1 (0.63) (day 2: 0.7153) (day 3: 0.7171)

# fit on the test data
lc_test_imp$prob <- predict(logit_bad_3, lc_test_imp, type = "response")

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group2_logit.csv", row.names = FALSE)


#####
#### decision tree
####

tree_data <- sample_frac(lc_train_imp, 0.1) # to fit faster (competition runs every day)

#lc_train_imp[1:100000, ] # just to test

library("caret")

ctrl <- trainControl( # create a caret trainControl object
  method = "cv", number = 1, # 3-fold CV
  selectionFunction = "best", # select the best performer
  classProbs = TRUE, # requested the predicted probs (for ROC)
  summaryFunction = twoClassSummary, # needed to produce the ROC/AUC measures
  savePredictions = TRUE # needed to plot the ROC curves
)

grid <- expand.grid(trials = 1,
                    winnow = FALSE,
                    model = "tree")

set.seed(0320)
m.c50 <- train(ranger_imp_formula, 
               data = lc_train_imp_small,
               method = "C5.0",
               trControl = ctrl,
               tuneGrid = grid,
               metric = "ROC")

roc.c50 <- roc(predictor = m.c50$pred$yes, response = m.c50$pred$obs)
plot(roc.c50, add = TRUE, color = "green")

auc.c50 <- round(auc(roc.c50), 3)

#####
#### random forest
####

library(ranger)
set.seed(1705)
m.ranger <- ranger(default ~ . -id, data = lc_train_imp_small,
                   importance = "impurity", probability = TRUE, 
                   num.trees = 500, mtry = 14)

# examine the importance, pull the names of the important ones (and add to a logit)
imp_ranger <- as_tibble(ranger::importance(m.ranger))
imp_ranger$names <- names(ranger::importance(m.ranger))

# create a formula for a new logit model (or any other model, really)
target = "default"
imp_vars <- imp_ranger %>% arrange(desc(value)) %>% filter(value > 15) %>% pull(names) 
ranger_imp_formula <- paste(target, '~', paste(imp_vars, collapse=' + ' ) )
ranger_imp_formula <- as.formula(ranger_imp_formula)

# do predictions within ranger 
pred_ranger <- predict(m.ranger, data = lc_val_imp_small)
lc_val_imp_small$preds_ranger <- as.numeric(pred_ranger$predictions[,2])

pred_ranger_roc <- roc(predictor = lc_val_imp_small$preds_ranger, response = lc_val_imp_small$default)
plot(pred_ranger_roc, main = "ROC curve for Lending Club Defaults", col = "red",
     lwd = 2, legacy.axes = TRUE, add = TRUE)
auc(pred_ranger_roc) # substantially better than day 1 (-) (day 2: -) (day 3: 200 trees:0.7141, 300 trees:0.7157, 400 trees: 0.7165, 500 trees: 0.7173, 500 trees, mtry = 14 (24 the same): 0.717 )

# fit on the test data
pred_ranger_test_data <- predict(m.ranger, data = lc_test_imp)
lc_test_imp$prob <- as.numeric(pred_ranger_test_data$predictions[,2])

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group2_RF.csv", row.names = FALSE)


#####
#### LOGIT: important vars from ranger
####

logit_imp_ranger <- glm(ranger_imp_formula,
                   data = lc_train_imp_small, family = "binomial") # garbage bin no more (at least less)
summary(logit_imp_ranger)

# in-model performance
lc_train_imp_small$preds_val_in_model <- predict(logit_imp_ranger, lc_train_imp_small, type = "response")

logit_imp_ranger_roc_in_model <- roc(predictor = lc_train_imp_small$preds_val_in_model, 
                                     response = lc_train_imp_small$default)
plot(logit_imp_ranger_roc_in_model, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(logit_imp_ranger_roc_in_model) # (day 5 = 0.7236)

# out-model performance
preds_val <- predict(logit_imp_ranger, lc_val_imp_small, type = "response")
preds_class <- ifelse(preds_val > 0.5, "pos", "neg")

lc_val_imp_small$preds_logit <- preds_val 
lc_val_imp_small$preds_class <- preds_class

prop.table(table(lc_val_imp_small$preds_class, lc_val_imp_small$default))

logit_imp_ranger_roc <- roc(predictor = lc_val_imp_small$preds_logit, response = lc_val_imp_small$default)
plot(logit_imp_ranger_roc, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(logit_imp_ranger_roc) # substantially better than day 1 (0.63) (day 2: 0.7153) (day 3: 0.7208) (day 4:0.7209, day 5: 0.722)

# fit on the test data
lc_test_imp$prob <- predict(logit_imp_ranger, lc_test_imp, type = "response")

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group2_logit_ranger.csv", row.names = FALSE)


### logit no not sig
logit_imp_ranger_no_sig <- glm(default ~ term + dti + loan_amnt + annual_inc + revol_util + revol_bal + mo_sin_old_rev_tl_op +
                                 bc_util + avg_cur_bal + total_bc_limit + tot_hi_cred_lim + total_bal_ex_mort +
                                 total_acc + mths_since_recent_bc + total_il_high_credit_limit + addr_state +
                                 acc_open_past_24mths + num_rev_accts + mo_sin_rcnt_rev_tl_op + mths_since_recent_inq +
                                 emp_length + mo_sin_rcnt_tl + num_bc_tl + num_bc_tl + mths_since_last_delinq + 
                                 percent_bc_gt_75 + pct_tl_nvr_dlq + num_tl_op_past_12m + num_rev_tl_bal_gt_0 +
                                 num_actv_rev_tl + purpose + mort_acc + inq_last_6mths + verification_status_joint + 
                                 mths_since_last_record + delinq_2yrs + earliest_cr_cat + pub_rec_bankruptcies +
                                 num_tl_90g_dpd_24m + mths_since_last_delinq_mvi + mths_since_recent_bc_dlq_mvi +
                                 mths_since_recent_inq_mvi + collections_12_mths_ex_med + mo_sin_old_il_acct_mvi +
                                 total_cu_tl,
                        data = lc_train_imp_small, family = "binomial") # garbage bin no more (at least less)
summary(logit_imp_ranger_no_sig)


preds_val_no_sig <- predict(logit_imp_ranger_no_sig, lc_val_imp_small, type = "response")

lc_val_imp_small$preds_logit_no_sig <- preds_val_no_sig 

logit_imp_ranger_no_sig_roc <- roc(predictor = lc_val_imp_small$preds_logit_no_sig, response = lc_val_imp_small$default)
plot(logit_imp_ranger_no_sig_roc, main = "ROC curve for Lending Club Defaults", col = "green",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE, add = TRUE)
auc(logit_imp_ranger_no_sig_roc)



#####
#### BOOST IT!
####

library(gbm)
set.seed(1705)
trees = 500

lc_train_imp_small_gbm <- lc_train_imp_small
lc_train_imp_small_gbm$default <- as.numeric(lc_train_imp_small_gbm$default)-1

lc_val_imp_small_gbm <- lc_val_imp_small
lc_val_imp_small_gbm$default <- as.numeric(lc_val_imp_small_gbm$default)-1

m_gbm <- gbm(default ~ . -id, # gbm did not run with these
             data = lc_train_imp_small_gbm,
             distribution = "bernoulli",
             n.trees = trees, # number of iterations
             interaction.depth = 3, # additive model (2 = 2-way, etc.) (probably at least 2)
             n.minobsinnode = 1, # affects overfitting
             shrinkage = 0.1, # values above/below this are more/less aggressive
             verbose = "CV")

m_gbm # examine the model(all 7 features important)
summary(m_gbm) # obtain relative feature importance


# in-model assessment
lc_train_imp_small_gbm$prob_gbm <- predict(m_gbm, lc_train_imp_small_gbm, n.trees = trees, type = "response")

gbm_roc_in_model <- roc(predictor = lc_train_imp_small_gbm$prob_gbm, response = lc_train_imp_small_gbm$default)

plot(gbm_roc_in_model, main = "ROC curve for Lending Club Defaults", col = "green",
                       lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(gbm_roc_in_model) # 500 trees: 0.7347 (400 trees, interaction:3: auc: .7377)

# obtain the predicted probability out-of-model
# note: you must specify the number of trees used for prediction!
lc_val_imp_small_gbm$prob_gbm <- predict(m_gbm, lc_val_imp_small_gbm, n.trees = trees, type = "response")
head(lc_val_imp_small_gbm$prob_gbm) # these will be probabilities

# convert the predicted probability into a prediction
lc_val_imp_small_gbm$pred_gbm <- ifelse(lc_val_imp_small_gbm$prob_gbm > 0.50, 1, 0)
lc_val_imp_small_gbm$pred_gbm # these will be 1/0


gbm_roc <- roc(predictor = lc_val_imp_small_gbm$prob_gbm, response = lc_val_imp_small_gbm$default)

plot(gbm_roc, main = "ROC curve for Lending Club Defaults", col = "green",
                       lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(gbm_roc) # 100 trees:0.7156, 300 trees: 0.7264, 400 trees: AUC.728, 500 trees: 07285, 400 trees + interact 3: 0.7297



#### 
#### SAMLING FOR ENSEMBLING

# divide full train into smaller train and validation sets
set.seed(1705) # set seed for replicability
# smaller train
tr_lc_ens <- sample_n(lc_train_imp, 250000) # randomly sample 50% of train data
ens_train <- lc_train_imp %>%  
  semi_join(tr_lc_ens, by = "id") 

# validation
lc_val_big_ens <- lc_train_imp %>% 
  anti_join(tr_lc_ens, by = "id") # save those not in train to validation set

set.seed(170501)
val_lc_ens <- sample_n(lc_val_big_ens, 125000)
ens_val_1 <- lc_val_big_ens %>%  
  semi_join(val_lc_ens, by = "id") 

ens_val_2 <- lc_val_big_ens %>% 
  anti_join(val_lc_ens, by = "id") # save those not in train to validation set

#### ENSEMBLE
## RANDOM FOREST
m_ens_ranger <- ranger(default ~ . -id, data = ens_train,
                   importance = "impurity", probability = TRUE, 
                   num.trees = 500, mtry = 14)

ranger::importance(m_ens_ranger)

## LOGIT 
m_ens_logit_imp_ranger <- glm(ranger_imp_formula,
                        data = ens_train, family = "binomial") # garbage bin no more (at least less)
summary(m_ens_logit_imp_ranger)

## do predictions on first validation set to create ranger_p1, logit_p_1
p_ens_ranger <- predict(m_ens_ranger, data = ens_val_1)
p_ens_ranger <- as.numeric(p_ens_ranger$predictions[,2])

p_ens_logit <- predict(m_ens_logit_imp_ranger,ens_val_1, type = "response")

roc_ens_logit <- roc(response = ens_val_1$default, predictor = p_ens_logit)
roc_ens_ranger <- roc(response = ens_val_1$default, predictor = p_ens_ranger)

auc_ens_logit <- round(auc(roc_ens_logit), 4)
auc_ens_ranger <- round(auc(roc_ens_ranger), 4)

## compare two models ROC curves
plot(roc_ens_logit, col = 1, lty = 2, main = "ROC")
plot(roc_ens_ranger, col = 4, lty = 3, add = TRUE)
legend("bottomright",
       legend = c(paste0("LR (", auc_ens_logit, ")"),
                  paste0("RF (", auc_ens_ranger, ")")),
       col = c("red", "blue"), lwd = 2)

## Create the 2nd Stage Model ----
# add predicted default to validation set
ens_val_1$p_ens_logit <- p_ens_logit
ens_val_1$p_ens_ranger <- p_ens_ranger

# stack the model
m_stack <- glm(default ~ p_ens_logit + p_ens_ranger,
               data = ens_val_1, family = binomial)

summary(m_stack) # see the results

# in-model ROC/AUC
pr_stack_in_model <- predict(m_stack, ens_val_1, type = "response")
roc_stack_in_model <- roc(response = ens_val_1$default, predictor = pr_stack_in_model)
auc(roc_stack_in_model) # Area under the curve: 0.7263 -- note: there may be overfitting!

# on second validation set
pr_stack_val_2 <- predict(m_stack, new_data = ens_val_2, type = "response")
roc_stack_val_2 <- roc(response = ens_val_2$default, predictor = pr_stack_val_2)
auc(roc_stack_val_2) # Area under the curve: 0.5009 -- note: this was not good!









####################################################################################
##################### EXPERIMENTS ##################################################
####################################################################################



#####
#### GBM MODELS EXPERIMENT
####
library(gbm)

# create data for gbm (takes no target factor by numeric)
train_gbm <- lc_train_imp_small
set.seed(1705)
#train_gbm$default <- ifelse(train_gbm$default == "yes", 1, 0)

val_gbm <- lc_val_imp_small 
#val_gbm$default <- ifelse(val_gbm$default == "yes", 1, 0)

# set up the ctrl
ctrl_test <- ctrl <- trainControl(method = "cv", 
                                  number = 3,
                                  selectionFunction = "best",
                                  classProbs = TRUE, # calculates probs alongside class type
                                  savePredictions = TRUE,
                                  summaryFunction = twoClassSummary) # gives ROC, sensitivity, specificity 

# what to tune
modelLookup("gbm")

# create experiment grid
grid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                    n.trees = seq(150, 550, by = 100),
                    shrinkage = c(0.05, 0.1),
                    n.minobsinnode = 10)
                    
set.seed(1705)
# the model (add train.fraction next)
gbm_tune <- train(default ~ . -id, 
                  data = train_gbm, 
                  method = "gbm",
                  metric = "ROC", # optimise this
                  tuneGrid = grid,
                  trControl = ctrl_test)

gbm_tune
gbm_tune$results
gbm_tune$bestTune
gbm_tune$finalModel
gbm_tune$pred

gbmImp <- varImp(gbm_tune, scale = TRUE) # get most important variables

png("test.png")
plot(gbm_tune, main = "Outcome experiment gbm")
dev.off()

roc_gbm_best <- roc(gbm_tune$pred$obs, gbm_tune$pred$yes)
png("best_tuned_gbm_roc.png")
plot(roc_gbm_best, main = "ROC curve for Lending Club Defaults", col = "green",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
dev.off()
auc(roc_gbm_best)

# obtain the predicted probability for the validation data (just as an extra and to practice)
# note: you must specify the number of trees used for prediction!
val_gbm$prob_gbm_tune <- predict(gbm_tune, newdata = val_gbm, type = "prob") # , n.trees = trees?
head(val_gbm$prob_gbm_tune) # these will be probabilities

val_gbm %>% select(default, prob_gbm_tune)

# roc best tune on validation data
roc_gbm_val_best <- roc(response = val_gbm$default, predictor = val_gbm$prob_gbm_tune$yes)
png("best_tune_gbm_on_val")
plot(roc_gbm_val_best, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
dev.off()
roc_gbm_val_best_auc <- round(auc(roc_gbm_val_best), 4)

# convert the predicted probability into a prediction (for confusion matrix)
val_gbm$pred_class_gbm <- ifelse(val_gbm$prob_gbm_tune$yes > 0.50, "yes", "no")
val_gbm$pred_class_gbm # these will be yes/no

confusionMatrix(as.factor(val_gbm$pred_class_gbm), val_gbm$default)

# predict on test data
pred_best_tune_gbm_test <- predict(gbm_tune, newdata = lc_test_imp, type = "prob")
lc_test_imp$prob <- pred_best_tune_gbm_test$yes

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group_2_gbm_best_tune.csv", row.names = FALSE)



##
## BEST RF
# create data for ranger
train_ranger <- lc_train_imp_small
train_ranger_small <- sample_frac(train_ranger, 0.33) # for faster computing
val_ranger <- lc_val_imp_small 

# what to tune
modelLookup("ranger")

# set-up
trees = 600
grid_rf <- expand.grid(mtry = c(12, 14),
                       splitrule = "gini",
                       min.node.size = 1)

# run the ranger experiment with caret
set.seed(1705)
ranger_tune <- train(default ~ . -id, 
                  data = train_ranger, 
                  method = "ranger",
                  metric = "ROC", # optimise this
                  tuneGrid = grid_rf,
                  trControl = ctrl_test,
                  importance = 'impurity')

# plot the outcome of the experiment
#png("ranger_exp_12_14.png")
plot(ranger_tune, main = "Outcome experiment: RF (ranger) with mtry 12 performs best on AUC")
#dev.off()

varImp(ranger_tune)
summary(ranger_tune)

# ROC/AUC from caret
roc_rf_tuned <- roc(ranger_tune$pred$obs, ranger_tune$pred$yes)
plot(roc_rf_tuned, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
roc_rf_tuned_auc <- auc(roc_rf_tuned)

confusionMatrix(ranger_tune$pred$pred, ranger_tune$pred$obs)

# predict with best model on validation data
pred_ranger <- predict(ranger_tune, newdata = val_ranger, type = "prob")
val_ranger$pred_ranger <- pred_ranger$yes

val_ranger %>% select(default, pred_ranger) %>% View()

m_rf_val_best_roc <- roc(predictor = val_ranger$pred_ranger, response = val_ranger$default)
plot(m_rf_val_best_roc, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
roc_rf_val_best_tuned_auc <- round(auc(m_rf_val_best_roc), 4)


# convert the predicted probability into a prediction
val_ranger$pred_class_rf <- ifelse(val_ranger$pred_ranger > 0.50, "yes", "no")
val_ranger$pred_class_rf # these will be 1/0

confusionMatrix(as.factor(val_ranger$pred_class_rf), as.factor(val_ranger$default))

# predict on test data
pred_best_tune_rf_test <- predict(ranger_tune, newdata = lc_test_imp, type = "prob")
lc_test_imp$prob <- pred_best_tune_rf_test$yes

# write CSV file to current directory
#write.csv(lc_test_imp[c("id", "prob")], "group_2_rf_best_tune.csv", row.names = FALSE)

### plot two best models on validation data
#png("roc_best_tune_gbm_rf_logit.png")
plot(roc_gbm_val_best, col = 1, lty = 2, main = "ROC-curves on validation data")
plot(m_rf_val_best_roc, col = 4, lty = 3, add = TRUE)
plot(roc_logit_caret_val, col = 8, lty = 4, add = TRUE)
legend("bottomright",
       legend = c(paste0("GBM best (", roc_gbm_val_best_auc, ")"),
                  paste0("Logit best (", roc_logit_caret_val_auc, ")"),
                  paste0("RF best (", roc_rf_val_best_tuned_auc, ")")),
       col = c("1", "4", "8"), lwd = 2)
#dev.off()


resamps <- resamples(list(GBM = gbm_tune,
                          RF = ranger_tune,
                          LOGIT = l_caret_cv_r_imp))

#png("resamps.png")
dotplot(resamps, metric = "ROC")
#dev.off()



#############################
#############################
##### MANUAL MODELS #########
#############################
#############################


# the best gbm model (manual in case something breaks above): 
m_gbm_best <- gbm(default ~ . -id, # gbm did not run with these
                  data = train_gbm,
                  distribution = "bernoulli",
                  n.trees = 550, # number of iterations
                  interaction.depth = 7, # additive model (2 = 2-way, etc.) (probably at least 2)
                  n.minobsinnode = 1, # affects overfitting
                  shrinkage = 0.1) # values above/below this are more/less aggressive)

summary(m_gbm_best) # variable importance

# on validation data
lc_val_imp_small_gbm$prob_gbm <- predict(m_gbm_best, lc_val_imp_small_gbm, n.trees = 550, type = "response")

m_gbm_best_roc <- roc(predictor = lc_val_imp_small_gbm$prob_gbm, response = lc_val_imp_small_gbm$default)

plot(m_gbm_best_roc, main = "ROC curve for Lending Club Defaults", col = "green",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
m_gbm_best_roc_auc <- round(auc(m_gbm_best_roc), 4) # 100 trees:0.7156, 300 trees: 0.7264, 400 trees: AUC.728, 500 trees: 07285, 400 trees + interact 3: 0.7297

# convert the predicted probability into a prediction
lc_val_imp_small_gbm$pred_gbm <- ifelse(lc_val_imp_small_gbm$prob_gbm > 0.50, 1, 0)
lc_val_imp_small_gbm$pred_gbm # these will be 1/0

confusionMatrix(as.factor(lc_val_imp_small_gbm$pred_gbm), as.factor(lc_val_imp_small_gbm$default))


# best gbm with caret
set.seed(1705)
# create data for gbm (takes no target factor by numeric)
train_gbm <- lc_train_imp_small
#train_gbm$default <- ifelse(train_gbm$default == "yes", 1, 0)
val_gbm <- lc_val_imp_small 
#val_gbm$default <- ifelse(val_gbm$default == "yes", 1, 0)

# set up the ctrl
ctrl_test <- ctrl <- trainControl(method = "cv", 
                                  number = 3,
                                  selectionFunction = "best",
                                  classProbs = TRUE, # calculates probs alongside class type
                                  savePredictions = TRUE,
                                  summaryFunction = twoClassSummary) # gives ROC, sensitivity, specificity 

# what to tune
modelLookup("gbm")

# create experiment grid
grid <- expand.grid(interaction.depth = 7,
                    n.trees = 550,
                    shrinkage = 0.1,
                    n.minobsinnode = 10)

set.seed(1705)
# the model (add train.fraction next)
gbm_tune <- train(default ~ . -id, 
                  data = train_gbm, 
                  method = "gbm",
                  metric = "ROC", # optimise this
                  tuneGrid = grid,
                  trControl = ctrl_test)


roc_gbm_best <- roc(gbm_tune$pred$obs, gbm_tune$pred$yes)
plot(roc_gbm_best, main = "ROC curve for Lending Club Defaults", col = "green",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(roc_gbm_best)

confusionMatrix(gbm_tune$pred$pred, gbm_tune$pred$obs)

# obtain the predicted probability for the validation data (just as an extra and to practice)
# note: you must specify the number of trees used for prediction!
val_gbm$prob_gbm_tune <- predict(gbm_tune, newdata = val_gbm, type = "prob") # , n.trees = trees?
head(val_gbm$prob_gbm_tune) # these will be probabilities

val_gbm %>% select(default, prob_gbm_tune)

# roc best tune on validation data
roc_gbm_val_best <- roc(response = val_gbm$default, predictor = val_gbm$prob_gbm_tune$yes)
plot(roc_gbm_val_best, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
roc_gbm_val_best_auc <- round(auc(roc_gbm_val_best), 4)

# convert the predicted probability into a prediction (for confusion matrix)
val_gbm$pred_class_gbm <- ifelse(val_gbm$prob_gbm_tune$yes > 0.50, "yes", "no")
val_gbm$pred_class_gbm # these will be yes/no

confusionMatrix(as.factor(val_gbm$pred_class_gbm), val_gbm$default)

val_gbm$prob_gbm_tune <- predict(gbm_tune, newdata = val_gbm, type = "prob") # , n.trees = trees?

# predict on test data
pred_best_tune_gbm_manual_test <- predict(gbm_tune, newdata = lc_test_imp, type = "prob")
lc_test_imp$prob <- pred_best_tune_gbm_manual_test$yes

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group_2_gbm_best_tune_manual.csv", row.names = FALSE)


## RF MANUAL

library(ranger)
set.seed(1705)
# 
ranger_best_for_imp <- ranger(default ~ . -id, data = train_ranger_small, # run with the small train data for speed
                   importance = "impurity", probability = TRUE, 
                   num.trees = 600, mtry = 12)

# examine the importance, pull the names of the important ones (and add to a logit)
imp_ranger_best <- as_tibble(ranger::importance(ranger_best_for_imp))
imp_ranger_best$names <- names(ranger::importance(ranger_best_for_imp))

# create a formula for a new logit model (or any other model, really)
target = "default"
imp_vars_rf_best <- imp_ranger_best %>% arrange(desc(value)) %>% filter(value > 15) %>% pull(names) 
ranger_imp_best_formula <- paste(target, '~', paste(imp_vars_rf_best, collapse=' + ' ) )
ranger_imp_best_formula <- as.formula(ranger_imp_best_formula)

# do predictions within ranger 
pred_ranger <- predict(ranger_best_for_imp, data = lc_val_imp_small)
lc_val_imp_small$preds_ranger <- as.numeric(pred_ranger$predictions[,2])

pred_ranger_best_roc <- roc(predictor = lc_val_imp_small$preds_ranger, response = lc_val_imp_small$default)
plot(pred_ranger_best_roc, main = "ROC curve for Lending Club Defaults", col = "red",
     lwd = 2, legacy.axes = TRUE)
auc(pred_ranger_best_roc) 

# fit on the test data
pred_ranger_test_data <- predict(ranger_best, data = lc_test_imp)
lc_test_imp$prob <- as.numeric(pred_ranger_test_data$predictions[,2])

# write CSV file to current directory
write.csv(lc_test_imp[c("id", "prob")], "group2_RF_best_manual.csv", row.names = FALSE)




#####
#### LOGIT: important vars from ranger
####

logit_imp_ranger <- glm(ranger_imp_best_formula,
                        data = lc_train_imp_small, family = "binomial") # garbage bin no more (at least less)
summary(logit_imp_ranger)

# in-model performance
lc_train_imp_small$preds_val_in_model <- predict(logit_imp_ranger, lc_train_imp_small, type = "response")

logit_imp_ranger_roc_in_model <- roc(predictor = lc_train_imp_small$preds_val_in_model, 
                                     response = lc_train_imp_small$default)
plot(logit_imp_ranger_roc_in_model, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(logit_imp_ranger_roc_in_model) # (day 5 = 0.7236)

# out-model performance
preds_val <- predict(logit_imp_ranger, lc_val_imp_small, type = "response")
preds_class <- ifelse(preds_val > 0.5, "pos", "neg")

lc_val_imp_small$preds_logit <- preds_val 
lc_val_imp_small$preds_class <- preds_class

prop.table(table(lc_val_imp_small$preds_class, lc_val_imp_small$default))

logit_imp_ranger_roc <- roc(predictor = lc_val_imp_small$preds_logit, response = lc_val_imp_small$default)
plot(logit_imp_ranger_roc, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
auc(logit_imp_ranger_roc) # substantially better than day 1 (0.63) (day 2: 0.7153) (day 3: 0.7208) (day 4:0.7209, day 5: 0.722)

###### LOGIT WITH CARET CV

logit_ctrl <- ctrl <- trainControl(method = "cv", 
                                   number = 3,
                                   selectionFunction = "best",
                                   classProbs = TRUE, # calculates probs alongside class type
                                   savePredictions = TRUE,
                                   summaryFunction = twoClassSummary) # gives ROC, sensitivity, specificity 

modelLookup("glm")

l_caret_cv_r_imp = train(
  form = ranger_imp_best_formula,
  data = lc_train_imp_small,
  trControl = logit_ctrl,
  method = "glm",
  family = "binomial")

# ROC/AUC from caret
roc_logit_caret <- roc(l_caret_cv_r_imp$pred$obs, l_caret_cv_r_imp$pred$yes)
plot(roc_logit_caret, main = "ROC curve for Lending Club Defaults", col = "blue",
     lwd = 2, legacy.axes = TRUE, print.auc = TRUE)
roc_logit_caret_auc <- auc(roc_logit_caret)

confusionMatrix(l_caret_cv_r_imp$pred$pred, l_caret_cv_r_imp$pred$obs)

# get predictions from caret object to validation data
l_caret_val_preds <- predict(l_caret_cv_r_imp, newdata = lc_val_imp_small, type = "prob")
lc_val_imp_small$preds_l_caret <- l_caret_val_preds$yes

roc_logit_caret_val <- roc(lc_val_imp_small$default, lc_val_imp_small$preds_l_caret)
plot(roc_logit_caret_val, print.auc =TRUE)
roc_logit_caret_val_auc <- round(auc(roc_logit_caret_val), 4)


save.image("save_all.RData")

save.image("save_small.RData")

