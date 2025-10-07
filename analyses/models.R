# Bayesian ordinal model for comparing LLMs on 5-point Likert outcomes across 11 domains
# UPDATED: assume each vignette is rated only once (no repeated raters or repeated ratings per vignette)
# ------------------------------------------------------------------------------
# Expected data frame `df` columns:
# - y: integer 1..5 (Likert outcome)
# - model: factor indicating LLM model (e.g., "gpt5", "gpt4", "llama")
# - domain: factor with 11 levels (domain names or codes)
# - vignette_id: (optional) factor for the vignette identifier — **only useful** if you want to model vignette-level variability and you have multiple vignettes per domain. If every vignette is unique and observed once, do NOT include vignette-level random effects.
# ------------------------------------------------------------------------------
# This script builds a hierarchical cumulative (ordinal) logistic model using brms,
# with partial pooling across domains and models and an interaction to allow
# model effects to vary by domain. It is simplified to reflect that each vignette
# is observed once (no rater random effects).


library(brms)
library(cmdstanr)
library(posterior)
library(tidyverse)
library(arrow)
library(ggthemes)
library(gt)
library(tidybayes)
library(purrr)

options(brms.backend = "cmdstanr")

readRenviron(".env")

data_folder=Sys.getenv("DATA_FOLDER")
output_folder=Sys.getenv("OUTPUT_FOLDER")


df=read_parquet(file.path(data_folder,"Combined review data.parquet"))
df <- df %>%
  mutate(
    y = as.integer(min_score),  # Ensure y is integer 1..5
    model = factor(model_name),
    domain = factor(dimension),
    vignette_id = factor(study_id),
    panel=factor(panel)
  ) %>%
  select(y, model, domain, vignette_id, study_id,panel) %>%
  filter(!is.na(y) )



# MODEL CHOICES (no rater random effect, no vignette repeated measures)
# -------------------------------------------------------------------
# Choice A (recommended when domains are the main grouping and you want partial pooling):
#   Allow model effects to partially pool across domains via random slopes by domain.
#   This borrows strength across domains while estimating domain-specific model effects.
formulaA <- bf(y ~ 1 + model + (1 + model | domain) + (1| panel))



# Priors (weakly informative) -- adjust to your context
priors <- c(
  prior(normal(0, 1.5), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 2), class = "Intercept"),           # global intercept
  prior(cauchy(0, 2), class = "sd"),          # SDs for group-level effects
  prior(lkj(2), class = "cor")                  # correlation prior for multivariate normal group effects
)

# Fit (example with formulaA). Adjust iter/warmup/chains for your compute.
fit <- brm(
  formula = formulaA,
  data = df,
  family = cumulative(link = "logit",threshold = "flexible"),
  prior = priors,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  seed = 2025,
  cores = parallel::detectCores()
)

# save the model to disk
#saveRDS(fit, file.path(output_folder,"brms_ordinal_model_fit2.rds"))
#fit=readRDS(file.path(output_folder,"brms_ordinal_model_fit2.rds"))

# Diagnostics and checks
post <- as_draws_df(fit)



print(fit)
pp_check(fit, type = "bars")
plot(fit)

fixef_summary <- fixef(fit, probs = c(0.025, 0.975)) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("Predictor") %>%
    filter(!grepl("Intercept", Predictor)) %>%
    mutate(
        OR = exp(Estimate),
        OR_low = exp(Q2.5),
        OR_high = exp(Q97.5)
    ) %>%
    select(Predictor, OR, OR_low, OR_high) %>%
    rename(
        `Odds Ratio` = OR,
        `2.5% CI` = OR_low,
        `97.5% CI` = OR_high
    ) %>% mutate(Predictor=recode(Predictor,"modeldeepSeekMr1"="DeepSeek-R1",
                                   "modelgeminiM2.5Mflash"="Gemini-2.5-Flash",
                                   "modelgptM4.1"="GPT4.1",
                                   "modelmedgemma"="MedGemma",
                                   "modelo3"="O3"))

# Print nicely formatted table

fixef_summary %>%
    gt() %>%
    fmt_number(columns = c(`Odds Ratio`, `2.5% CI`, `97.5% CI`), decimals = 2) %>%
    tab_header(title = "Odds Ratios for Fixed Effects (Model)") %>%
    gtsave(file.path(output_folder, "fixef_summary_table.html"))


newdat <- expand.grid(
model = unique(df$model),
domain = unique(df$domain),
panel=unique(df$panel)
)


pp <- add_epred_draws(fit, newdata = newdat, re_formula = NULL) %>%
mutate(high_score = .category %in% c("4", "5")) %>%
group_by(model, domain,panel, .draw) %>%
summarise(prob = sum(.epred[high_score]), .groups = "drop") %>%
group_by(model, domain) %>%
median_qi(prob, .width = 0.95)

pp<-pp %>% mutate(Domain=recode(domain,
align = "Alignment to medical guidelines",
clear_comm = "Clear communication",
context = "Contextual relevance",
dem_bias = "Demographic and \nsocio-economic bias",
extent_of_harm = "Severity of harm",
knowledge = "Expert-level knowledge base",
likelihood_harm = "Likelihood of harm",
logical = "Logical consistency",
omission = "Omission of critical\n information",
relevent = "Relevance to the\n question",
understand = "Understanding of the question")
,
Model=recode(model,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="O3","clinician"="Clinician"))


ggplot(pp, aes(x = Model, y = prob)) +
geom_point()+
geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0.2)+
facet_wrap(~Domain)+
ylab("P(rating >= 4)")+
xlab("Model")+
theme_igray()+
theme(axis.text.x = element_text(angle = 45, hjust = 1,face='bold'),
axis.title = element_text(face='bold'))

ggsave(file.path(output_folder,"Model_prob_ge_4_by_domain.png"),width=14,height=10,dpi=300)






# Posterior extraction and pairwise comparisons


# Pairwise contrast function (domain-wise) adapted for the random-slope parameterization
pairwise_diff_by_domain <- function(fit, modelA, modelB,baseline_model="clinician"){
  draws <- as_draws_df(fit)
  domains <- unique(df$domain)
  results <- list()
  if(modelA==baseline_model) modelA="Intercept"
  if(modelB==baseline_model) modelB="Intercept"

  for (d in domains){
    fixedA <- paste0("b_model", gsub("-","M",modelA))
    fixedB <- paste0("b_model", gsub("-","M",modelB))
    if (modelA=="Intercept"){
        randA <- paste0("r_domain[", d, ",", gsub("-","M",modelA), "]")
        }else{
            randA=paste0("r_domain[", d, ",model", gsub("-","M",modelA), "]")
        }
    if (modelB=="Intercept"){
        randB <- paste0("r_domain[", d, ",", gsub("-","M",modelB), "]")
        }else{  
    randB <- paste0("r_domain[", d, ",model", gsub("-","M",modelB), "]")}
    

    hasFixedA <- fixedA %in% colnames(draws)
    hasFixedB <- fixedB %in% colnames(draws)
    hasRandA <- randA %in% colnames(draws)
    hasRandB <- randB %in% colnames(draws)

    drawA <- 0
    drawB <- 0
    if (hasFixedA) drawA <- drawA + draws[[fixedA]]
    if (hasFixedB) drawB <- drawB + draws[[fixedB]]
    if (hasRandA) drawA <- drawA + draws[[randA]]
    if (hasRandB) drawB <- drawB + draws[[randB]]

    diff <- drawB-drawA
    results[[d]] <- tibble(domain = d,
                           modelA = modelA,
                           modelB = modelB,
                           median_diff = median(diff),
                           mean_diff = mean(diff),
                           prob_B_gt_A = mean(diff > 0),
                           conf_low = quantile(diff, 0.025),
                           conf_high = quantile(diff, 0.975))
  }
  bind_rows(results)
}



# Generate pairwise comparisons for all model pairs
models <- levels(df$model)
all_pairs <- combn(models, 2, simplify = FALSE)
pairwise_results <- map_dfr(all_pairs, ~ pairwise_diff_by_domain(fit, .x[1], .x[2]))
pairwise_results<-pairwise_results %>% mutate(modelA=if_else(modelA=="Intercept","clinician",modelA),
                                              modelB=if_else(modelB=="Intercept","clinician",modelB)) %>% 
                                              mutate(modelA=recode(modelA,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="O3","clinician"="Clinician"),
                                   modelB=recode(modelB,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="O3","clinician"="Clinician"),
                                   comparison= sprintf("%s vs %s", modelB, modelA),
                                   Domain=recode(domain,
align = "Alignment to medical guidelines",
clear_comm = "Clear communication",
context = "Contextual relevance",
dem_bias = "Demographic and \nsocio-economic bias",
extent_of_harm = "Severity of harm",
knowledge = "Expert-level knowledge base",
likelihood_harm = "Likelihood of harm",
logical = "Logical consistency",
omission = "Omission of critical\n information",
relevent = "Relevance to the\n question",
understand = "Understanding of the question")) %>% 
                                   mutate(or_ci=sprintf("%.2f (%.2f, %.2f)",exp(median_diff),exp(conf_low),exp(conf_high)))

ggplot(pairwise_results)+
    geom_point(aes(x=reorder(comparison,modelA),y=median_diff))+
    geom_errorbar(aes(x=reorder(comparison,modelA),ymin=conf_low,ymax=conf_high),width=0.2)+
    geom_hline(yintercept=0, linetype="dashed", color = "red")+
    facet_wrap(~Domain)+
ylab("Difference in log-odds (Model B - Model A)")+
xlab("Model Comparison")+
theme_igray()+
theme(axis.text.x = element_text(angle = 90, hjust = 1,face='bold'),
axis.title = element_text(face='bold')) 

ggsave(file.path(output_folder,"Pairwise_model_differences_by_domain.png"),width=14,height=10,dpi=300)





# Summarizes posterior predicted probabilities for each model/domain/category combination.
# - fit: brmsfit object
# - newdat: data.frame of new observations (rows preserved)
# - re_formula: passed to brms::add_epred_draws (use NULL to include group-level effects)
# - draws: number of posterior draws to sample (NULL uses all draws)
#
# Returns: tibble with columns model, domain, .category, and the median and 95% credible interval
#          (lower, upper) for the predicted probability in each category.

posterior_predictive <- function(fit, newdat, re_formula = NULL, draws = NULL) {
  fit %>% 
    add_epred_draws(newdata = newdat, re_formula = re_formula, draws = draws) %>%
    rename(y_pred = .epred) %>%
    mutate(obs = row_number()) %>%
    select(.draw, obs, y_pred, everything()) -> pred_draws

  pred_draws %>% 
    group_by(model, domain, .category) %>% 
    summarise(
      median = median(y_pred),
      lower = quantile(y_pred, 0.025),
      upper = quantile(y_pred, 0.975)
    ) -> pred_summary

  return(pred_summary)
}


#posterior predictive checks
posterior_prediction_fit<-posterior_predictive(fit, newdat, re_formula = NULL)
observed_probs<-df %>% group_by(model, domain, y) %>% summarise(n=n()) %>%
  group_by(model, domain) %>% mutate(obs_prob=n/sum(n),y=as.character(y)) %>%
  ungroup() %>% select(-n) 

observed_pred_probs<-left_join(observed_probs,posterior_prediction_fit,
                          by=c("model"="model","domain"="domain","y"=".category"))


pdf(file.path(output_folder,"posterior_predictive_checks.pdf"),width=14,height=10)
for( d in unique(observed_pred_probs$domain)){
  p<-ggplot(observed_pred_probs %>% filter(domain==d),aes(x=y))+
    geom_point(aes(y=obs_prob,color="Observed"),size=3)+
    geom_point(aes(y=median,color="Predicted"),shape=17,size=3)+
    geom_errorbar(aes(ymin=lower,ymax=upper,color="Predicted"),width=0.2)+
    facet_wrap(~model)+
    ylab("Probability")+
    xlab("Rating Category")+
    ggtitle(paste("Posterior Predictive Check - Domain:",d))+
    scale_color_manual(name="",values=c("Observed"="black","Predicted"="blue"))+
    theme_igray()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1,face='bold'),
          axis.title = element_text(face='bold'),
          plot.title = element_text(face='bold',hjust=0.5))
  print(p)
}
dev.off()








# Sensitivity analyses: formulaA with non informative priors
# Priors (non-informative) -- adjust to your context
priors_ni <- c(
  prior(normal(0, 10), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 10), class = "Intercept"),           # global intercept
  prior(exponential(0.5), class = "sd"),          # SDs for group-level effects
  prior(lkj(1), class = "cor")                  # correlation prior for multivariate normal group effects
) 

# Fit (example with formulaA). Adjust iter/warmup/chains for your compute.
fit_ni <- brm(
  formula = formulaA,
  data = df,
  family = cumulative(link = "logit",threshold = "flexible"),
  prior = priors_ni,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  seed = 2025,
  cores = parallel::detectCores()
)

#save the model to disk
#saveRDS(fit_ni, file.path(output_folder,"brms_ordinal_model_fit_ni.rds"))
#fit_ni=readRDS(file.path(output_folder,"brms_ordinal_model_fit_ni.rds"))


# posterior predictive checks
posterior_prediction_fit_ni<-posterior_predictive(fit_ni, newdat, re_formula = NULL)
observed_probs<-df %>% group_by(model, domain, y) %>% summarise(n=n()) %>%
  group_by(model, domain) %>% mutate(obs_prob=n/sum(n),y=as.character(y)) %>%
  ungroup() %>% select(-n) 

observed_pred_probs_ni<-left_join(observed_probs,posterior_prediction_fit_ni,
                          by=c("model"="model","domain"="domain","y"=".category"))


pdf(file.path(output_folder,"posterior_predictive_checks_non informative.pdf"),width=14,height=10)
for( d in unique(observed_pred_probs_ni$domain)){
  p<-ggplot(observed_pred_probs_ni %>% filter(domain==d),aes(x=y))+
    geom_point(aes(y=obs_prob,color="Observed"),size=3)+
    geom_point(aes(y=median,color="Predicted"),shape=17,size=3)+
    geom_errorbar(aes(ymin=lower,ymax=upper,color="Predicted"),width=0.2)+
    facet_wrap(~model)+
    ylab("Probability")+
    xlab("Rating Category")+
    ggtitle(paste("Posterior Predictive Check - Domain:",d))+
    scale_color_manual(name="",values=c("Observed"="black","Predicted"="blue"))+
    theme_igray()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1,face='bold'),
          axis.title = element_text(face='bold'),
          plot.title = element_text(face='bold',hjust=0.5))
  print(p)
}
dev.off()


## model with interaction between model and domain and random effects for panel
formulaB <- bf(y ~ 1  + model * domain + (1 | panel))


# Priors (non-informative) -- adjust to your context
priors_interactions <- c(
  prior(normal(0, 10), class = "b"),                # coefficients on latent (logit) scale
  prior(normal(0, 10), class = "Intercept"),           # global intercept
  prior(exponential(0.5), class = "sd")          # SDs for group-level effects
) 

# Fit (example with formulaA). Adjust iter/warmup/chains for your compute.
fit_interactions <- brm(
  formula = formulaB,
  data = df,
  family = cumulative(link = "logit",threshold = "flexible"),
  prior = priors_interactions,
  chains = 4,
  iter = 5000,
  warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  seed = 2025,
  cores = parallel::detectCores()
)

saveRDS(fit_interactions, file.path(output_folder,"brms_ordinal_model_fit_interactions.rds"))
#fit_interactions=readRDS(file.path(output_folder,"brms_ordinal_model_fit_interactions.rds"))


pairwise_model_comparisons_interaction <- function(fit, df) {

  
  models <- levels(df$model)
  domains <- levels(df$domain)
  # panels <- levels(df$panel)

  newdat <- expand.grid(model = models, domain = domains, stringsAsFactors = FALSE) %>%
    mutate(model = factor(model, levels = models),
           domain = factor(domain, levels = domains))

  # Get posterior draws for each model/domain combination
  lp_draws <- fit %>%
    add_linpred_draws(newdata = newdat, re_formula = NA, allow_new_levels = TRUE) %>%
    rename(lp = .linpred) %>%
    select(.draw, model, domain, lp)
  
  # Compute pairwise differences for each domain
  pairs <- combn(models, 2, simplify = FALSE)
  diff_draws <- map_dfr(pairs, function(pair) {
    A <- pair[1]; B <- pair[2]
    lp_A <- lp_draws %>% filter(model == A) %>% rename(lp_A = lp)
    lp_B <- lp_draws %>% filter(model == B) %>% rename(lp_B = lp)
    inner_join(lp_A, lp_B, by = c(".draw", "domain")) %>%
      transmute(.draw, domain, modelA = A, modelB = B,
                lp_diff = lp_B - lp_A,
                or = exp(lp_diff))
  })
  
  # Summarize
  summary_tbl <- diff_draws %>%
    group_by(domain, modelA, modelB) %>%
    summarise(
      median_diff = median(lp_diff),
      conf_low = quantile(lp_diff, 0.025),
      conf_high = quantile(lp_diff, 0.975),
      prob_B_gt_A = mean(lp_diff > 0),
      median_or = median(or),
      or_low = quantile(or, 0.025),
      or_high = quantile(or, 0.975),
      .groups = "drop"
    ) %>%
    arrange(domain, modelA, modelB)
  
  summary_tbl
}


pairwise_interaction=pairwise_model_comparisons_interaction(fit_interactions, df)

pairwise_interaction<-pairwise_interaction %>% 
                                              mutate(modelA=recode(modelA,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="o3","clinician"="Clinician"),
                                   modelB=recode(modelB,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="o3","clinician"="Clinician"),
                                   comparison= sprintf("%s vs %s", modelB, modelA),
                                   Domain=recode(domain,
align = "Alignment to medical guidelines",
clear_comm = "Clear communication",
context = "Contextual relevance",
dem_bias = "Demographic and \nsocio-economic bias",
extent_of_harm = "Severity of harm",
knowledge = "Expert-level knowledge base",
likelihood_harm = "Likelihood of harm",
logical = "Logical consistency",
omission = "Omission of critical\n information",
relevent = "Relevance to the\n question",
understand = "Understanding of the question")) %>% 
                                   mutate(or_ci=sprintf("%.2f (%.2f, %.2f)",exp(median_diff),exp(conf_low),exp(conf_high)))

ggplot(pairwise_interaction)+
    geom_point(aes(x=reorder(comparison,modelB),y=median_diff))+
    geom_errorbar(aes(x=reorder(comparison,modelB),ymin=conf_low,ymax=conf_high),width=0.2)+
    geom_hline(yintercept=0, linetype="dashed", color = "red")+
    facet_wrap(~Domain)+
ylab("Difference in log-odds")+
xlab("Model Comparison")+
theme_igray()+
theme(axis.text.x = element_text(angle = 90, hjust = 1,face='bold'),
axis.title = element_text(face='bold')) 

ggsave(file.path(output_folder,"Pairwise_model_differences_by_domain_interactions.png"),width=14,height=10,dpi=300)



newdat_interaction <- expand.grid(
model = unique(df$model),
domain = unique(df$domain)
)
posterior_prediction_interaction<-posterior_predictive(fit_interactions, newdat_interaction, re_formula = NA)
observed_probs<-df %>% group_by(model, domain, y) %>% summarise(n=n()) %>%
  group_by(model, domain) %>% mutate(obs_prob=n/sum(n),y=as.character(y)) %>%
  ungroup() %>% select(-n) 

observed_pred_probs_interactions<-left_join(observed_probs,posterior_prediction_interaction,
                          by=c("model"="model","domain"="domain","y"=".category"))


pdf(file.path(output_folder,"posterior_predictive_checks_interactions.pdf"),width=14,height=10)
for( d in unique(observed_pred_probs_interactions$domain)){
  p<-ggplot(observed_pred_probs_interactions %>% filter(domain==d),aes(x=y))+
    geom_point(aes(y=obs_prob,color="Observed"),size=3)+
    geom_point(aes(y=median,color="Predicted"),shape=17,size=3)+
    geom_errorbar(aes(ymin=lower,ymax=upper,color="Predicted"),width=0.2)+
    facet_wrap(~model)+
    ylab("Probability")+
    xlab("Rating Category")+
    ggtitle(paste("Posterior Predictive Check - Domain:",d))+
    scale_color_manual(name="",values=c("Observed"="black","Predicted"="blue"))+
    theme_igray()+
    theme(axis.text.x = element_text(angle = 45, hjust = 1,face='bold'),
          axis.title = element_text(face='bold'),
          plot.title = element_text(face='bold',hjust=0.5))
  print(p)
}
dev.off()


# propability of rating >=4
pp_interaction <- add_epred_draws(fit_interactions, newdata = newdat_interaction, re_formula = NA) %>%
mutate(high_score = .category %in% c("4", "5")) %>%
group_by(model, domain, .draw) %>%
summarise(prob = sum(.epred[high_score]), .groups = "drop") %>%
group_by(model, domain) %>%
median_qi(prob, .width = 0.95)  

pp_interaction <- pp_interaction %>% mutate(Domain=recode(domain,
align = "Alignment to medical guidelines",
clear_comm = "Clear communication",
context = "Contextual relevance",
dem_bias = "Demographic and \nsocio-economic bias",
extent_of_harm = "Severity of harm",
knowledge = "Expert-level knowledge base",
likelihood_harm = "Likelihood of harm",
logical = "Logical consistency",
omission = "Omission of critical\n information",
relevent = "Relevance to the\n question",
understand = "Understanding of the question")
,
Model=recode(model,"deepSeek-r1"="DeepSeek-R1",
                                   "gemini-2.5-flash"="Gemini-2.5-Flash",
                                   "gpt-4.1"="GPT4.1",
                                   "medgemma"="MedGemma",
                                   "o3"="O3","clinician"="Clinician"))


ggplot(pp_interaction, aes(x = Model, y = prob)) +
geom_point()+
geom_errorbar(aes(ymin = .lower, ymax = .upper), width = 0.2)+
facet_wrap(~Domain)+
ylab("P(rating >= 4)")+
xlab("Model")+
theme_igray()+
theme(axis.text.x = element_text(angle = 45, hjust = 1,face='bold'),
axis.title = element_text(face='bold'))

ggsave(file.path(output_folder,"Model_prob_ge_4_by_domain_interactions.png"),width=14,height=10,dpi=300)

