#Taka frá djúpkarfa og gera samskonar töflu með honum

#Lesa inn ný gögn

age_length_keys <- read_csv("C:/Ráðgjöf/djupkarfa_gullkarfa_ysa_2023/age_length_keys.csv")

length_dist_SMB <- read_csv("C:/Ráðgjöf/djupkarfa_gullkarfa_ysa_2023/length_dist_SMB.csv")

length_dist_SMH <- read_csv("C:/Ráðgjöf/djupkarfa_gullkarfa_ysa_2023/length_dist_SMH.csv")

length_dist_commercial <- read_csv("C:/Ráðgjöf/djupkarfa_gullkarfa_ysa_2023/length_dist_commercial.csv")


#Brjóta niður í smærri töflur/tegund

deepredfish_age_length_keys <-age_length_keys  %>% 
#  filter(ar>=2015)  %>% 
  filter(enskt_heiti=='Deepwater redfish')

deepredfish_length_dist_SMB<-length_dist_SMB %>% 
#  filter(ar>=2015)  %>% 
  filter(enskt_heiti=='Deepwater redfish')

deepredfish_length_dist_SMH<-length_dist_SMH %>% 
#  filter(ar>=2015)  %>% 
  filter(enskt_heiti=='Deepwater redfish')

#deepredfish_length_dist_commercial<-length_dist_commercial %>% 
#  filter(ar>=2016)  
#filter(enskt_heiti=='Deepwater redfish')
#Útgerðin telur ekki djúpkarfa virðist vera  









#búa til hnit fyrir deepwater redfish

deephnit<-
  deepredfish_length_dist_SMH %>%
  mutate(NE = case_when(
    lon > -18.5 & lat >65.4 ~ 1,
    TRUE ~ 0)) %>%
  mutate(NW = case_when(
    lon < -18.5 & lat >65.4 ~ 1,
    TRUE ~ 0)) %>%
  mutate(SE = case_when(
    lon > -18.5 & lat <65.4 ~ 1,
    TRUE ~ 0)) %>%
  mutate(SW = case_when(
    lon < -18.5 & lat <65.4 ~ 1,
    TRUE ~ 0))


#Skoða fjölda mismunandi tegunda í commercial töflunni... sýnist djúpkarfi ekki vera talinn

#deepredfish_length_dist_commercial%>%
#  count(heiti)

#deepredfish_length_dist_commercial%>%
#  count(enskt_heiti)



#Búa til töflu fyrir afla/landsfjórðung ásamt heildar afla 

#vegið hitastig

deep_weighted_temp_by_year<-deephnit %>%
  group_by(ar,lengd)%>%
  mutate(weighted_temp=weighted.mean(bottom_temp,fjoldi=maeldir+taldir),fjoldi=maeldir+taldir)


#búa til quantile table og per_length (fjöldi per lengd fyrir hvert ár)

deep_quantile_table <- deep_weighted_temp_by_year %>% 
  group_by (ar, station) %>% 
  summarise(sum_fjoldi=sum(fjoldi))%>%
  mutate(max_fjoldi=max(sum_fjoldi),
         quantile=quantile(sum_fjoldi,0.75, na.rm = TRUE))
  

deep_sumfjoldi_ar<-deep_weighted_temp_by_year %>% 
  group_by (ar) %>% 
  summarise(sum_fjoldi_ar=sum(fjoldi))

deep_sumfjoldi_lengd<-deep_weighted_temp_by_year %>% 
  group_by (ar,lengd) %>% 
  summarise(sum_fjoldi_lengd=sum(fjoldi))

###reikna per_length með hlutfallslegan fjölda/lengd fyrir hvert ár
deep_sumfjoldi <- deep_sumfjoldi_lengd%>% 
  left_join(deep_sumfjoldi_ar) %>% 
  group_by (ar) %>% 
  reframe( ar, lengd, per_length = sum_fjoldi_lengd/sum_fjoldi_ar )



#Færa fjölda yfir á landshluta, en halda einum dálk fyrir fjölda líka

deep_distribution <- deep_weighted_temp_by_year%>%
  group_by(ar,lengd)%>%
  dplyr::select(ar, station, lengd, weighted_temp, NW, NE, SW, SE, fjoldi)%>% 
  mutate(NW = case_when(NW == 1 ~ fjoldi, TRUE ~ 0 ),
         NE = case_when(NE == 1 ~ fjoldi, TRUE ~ 0 ),
         SW = case_when(SW == 1 ~ fjoldi, TRUE ~ 0 ),
         SE = case_when(SE == 1 ~ fjoldi, TRUE ~ 0 ))%>%
  arrange(ar, lengd)

temp_summary<-deep_weighted_temp_by_year%>%
  group_by (ar,lengd)%>%
  summarize(temp=mean(weighted_temp))
  
  
  
#Búa til distribution töflu með landshlutum, hitastigi etc

deep_distribution%>% 
  left_join(deep_quantile_table) %>%
  mutate(fjoldi=ifelse(sum_fjoldi<=quantile,fjoldi,fjoldi*quantile/sum_fjoldi)) %>%
  group_by(ar,lengd) %>% 
  summarise(sum_fjoldi=sum(fjoldi)) %>%
  left_join(deep_sumfjoldi) %>%
  arrange(ar,lengd) %>% 
  reframe(ar, lengd, bottom_temp, NW, NE, SW, SE, fjoldi, sum_fjoldi,cum=cumsum(sum_fjoldi), max(cum), per=cum/max(cum),per_length) %>% 
  write.csv("C:/Ráðgjöf/Maris Optimum/distributionoutput/deep_distribution.csv", row.names = FALSE)

#Búa til töflu sem líkist gömlu distribution + fractile töflunum... sendi beint í gr_data möppuna

deep_distribution%>% 
  left_join(deep_quantile_table) %>%
  left_join(deep_sumfjoldi) %>%
  mutate(fjoldi=ifelse(sum_fjoldi<=quantile,fjoldi,fjoldi*quantile/sum_fjoldi)) %>%
  group_by(ar,lengd) %>% 
  summarise(sum_fjoldi=sum(fjoldi)) %>%
  left_join(deep_sumfjoldi) %>%
  left_join(temp_summary) %>%
  arrange(ar,lengd) %>%
  reframe(ar, lengd, sum_fjoldi,cum=cumsum(sum_fjoldi), max(cum), per_length, per=cum/max(cum),temp) %>% 
  write.csv("C:/Ráðgjöf/Maris Optimum/gr_data/distribution101.csv", row.names = FALSE)



distribution101<-deep_distribution%>% 
  left_join(deep_quantile_table) %>%
  left_join(deep_sumfjoldi) %>%
  mutate(fjoldi=ifelse(sum_fjoldi<=quantile,fjoldi,fjoldi*quantile/sum_fjoldi)) %>%
  group_by(ar,lengd) %>% 
  summarise(sum_fjoldi=sum(fjoldi)) %>%
  left_join(deep_sumfjoldi) %>%
  arrange(ar,lengd) %>%
  reframe(ar, lengd, sum_fjoldi,cum=cumsum(sum_fjoldi), max(cum), per_length, per=cum/max(cum))