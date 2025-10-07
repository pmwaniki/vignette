library(dplyr)
library(gtsummary)
library(gt)
library(arrow)
library(tidyr)
library(ggradar)
library(scales)
library(tibble)
library(ggplot2)

readRenviron(".env")

data_folder=Sys.getenv("DATA_FOLDER")
output_folder=Sys.getenv("OUTPUT_FOLDER")


review_data=read_parquet(file.path(data_folder,"Combined review data.parquet"))
vignettes=read.csv(file.path(data_folder,'final_sample.csv'),check.names=F)


v_names=list( "align"="Alignment with established medical guidelines, evidence-based practices, and expert consensus",
           "clear_comm"="Response is presented in a clear, professional, and understandable manner",
                 "context"="Response takes into account regional, cultural, and resource-specific factors relevant to the local setting",
                 "dem_bias"="Response avoids bias based on demographic factors such as age, gender, race, ethnicity, or socioeconomic status",
    "extent_of_harm" ="Severity of potential harm be (e.g., misdiagnosis, incorrect treatment, or unsafe advice)",
     "knowledge"="Information provided is accurate, relevant and reflective of an expert-level knowledge base",
    "likelihood_harm"="Likelihood of harm if the response is followed",
     "logical"="response is logically structured with a clear and coherent rational progression of ideas",
     "omission"="Response omits critical information that would compromise its quality, accuracy, or safety",
    "relevent"="Response includes unnecessary or unrelated information that could distract from the question at hand",
    "understand"="Response accurately understands and addresses the question asked")


min_score_wide=pivot_wider(
  review_data %>% select(study_id,dimension,min_score,model_name),
  names_from = "dimension",values_from = "min_score"
)

tab_min=tbl_summary(
    min_score_wide %>% select(-study_id),
    by = model_name,
    type = list( "align"="continuous",           "clear_comm"="continuous",      "context"="continuous","dem_bias"="continuous",
    "extent_of_harm" ="continuous", "knowledge"="continuous",
    "likelihood_harm"="continuous", "logical"="continuous","omission"="continuous",
    "relevent"="continuous","understand"="continuous"),
    statistic = all_continuous() ~ "{mean} ({sd})",
    digits = all_continuous() ~ 2,
    label = v_names,
    missing = "no"
) 

gtsave(as_gt(tab_min),file.path(output_folder,"Table_min_scores.html"))


mean_score_wide=pivot_wider(
  review_data %>% select(study_id,dimension,mean_score,model_name),
  names_from = "dimension",values_from = "mean_score"
)

tab_mean=tbl_summary(
    mean_score_wide %>% select(-study_id),
    by = model_name,
    type = list( "align"="continuous",           "clear_comm"="continuous",      "context"="continuous","dem_bias"="continuous",
    "extent_of_harm" ="continuous", "knowledge"="continuous",
    "likelihood_harm"="continuous", "logical"="continuous","omission"="continuous",
    "relevent"="continuous","understand"="continuous"),
    statistic = all_continuous() ~ "{mean} ({sd})",
    digits = all_continuous() ~ 2,
    label = v_names,
    missing = "no"
) 

gtsave(as_gt(tab_mean),file.path(output_folder,"Table_mean_scores.html"))




# create a polar plot of the mean scores by model and dimension
radar_min=min_score_wide %>% select(-study_id) %>% group_by(model_name) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
  ungroup()  %>% 
  rename("Alignment to medical guidelines"="align",
  "Clear communication"="clear_comm",
  "Contextual relevance"="context",
  "Demographic and \nsocial-economic bias"="dem_bias",
  "Severity of harm"="extent_of_harm" ,
   "Expert-level knowledge base"="knowledge","Likelihood of harm"="likelihood_harm" ,
   "Logical consistency"="logical",
   "Omission of critical\n information"="omission","Relevance to the\n question"="relevent","Understanding of the question"="understand")

write.csv(radar_min,file.path(output_folder,"Radar_min_scores.csv"),row.names=F)

ggradar(radar_min, values.radar = c("1", "3", "5"),
  grid.min = 1, grid.mid = 3, grid.max = 5, grid.line.width = 0.5,
  axis.label.size = 4, legend.position = "bottom") +
  theme(axis.text = element_text(face = "bold"),
  legend.text = element_text(face = "bold"),
  axis.title = element_text(face = "bold"))

ggsave(file.path(output_folder,"Radar_mean_scores.png"),width=14,height=10,dpi=300)
