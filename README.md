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


FeaturePrep0313.py - extract date, time, media type & url; merge engagements from original data
	creates PrepData0313, dataset0313.csv
	total #obs 2093

