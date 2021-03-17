# master-modeler2021


This project is for Master Modeler Competition 2021, aiming to help ERASE Child Trafficking increase social media exposure.


DataCollection.py - GraphAPI
unavailable_urls.txt - url not working (error100, including deleted, not including repost)
RawData - JSON, index==excel order, including error100=>empty JSON
FilteredData == dataset.csv
DataCleaning.py - created_times=>pandas extraction; 


media_type summary
[share]                                 1267	=> [share]
[photo]                                  538	=> [photo]
[video_inline]                           207	=> [video]
[album]                                   21	=> [photo]
[]                                        21	ignore
[video_direct_response]                   17	=> [video]
[share, fundraiser_for_story]              6	ignore
[native_templates]                         4	=> [video], but dont have url
[cover_photo]                              3	delete, update cover photo
[fundraiser_for_story]                     2	ignore
[photo, fundraiser_for_story]              2	ignore
[avatar]                                   2	delete, others page
[visual_poll]                              2	ignore
[video_inline, fundraiser_for_story]       2 	ignore fundraiser_for_story
[video]                                    2	=> [video]
[map]                                      1	=> [video] video at a place, cannot get link to video
[profile_media]                            1    delete, update profile picture
[new_album]                                1    => [photo], photo not shown
^[link]											=> [link], added based on 'urls' column


FeaturePrep0313.py - Extract date, time, media type & url; merge engagements from original data
	create PrepData0313, dataset0313.csv
	total #obs 2093


FeaturePrep0314.py - Use OpenCV to extract image feature from photo and thumbnails
	create json_face, json_tn_face


FeaturePrep0315.py - Incorporate face features into 0313 data
	create PrepData0315, dataset0315.csv


PrelimAnalysis
	engagement_rate 16 outlier removed
	Quantile after outlier removed: 
		0.10     13.592867
		0.25     27.842227
		0.50     54.962819
		0.75    108.173077
		0.90    214.062232
		Name: engagement_rate

	total_engagement 13 outlier removed
	Quantile after outlier removed: 
		0.10     3.0
		0.25     6.0
		0.50    13.0
		0.75    28.0
		0.90    62.0
		Name: total_engagement
