import turicreate as tc

model = tc.load_model('/Users/yasnara/Desktop/temp.model')


test_data = tc.image_analysis.load_images('/Users/yasnara/Desktop/a', with_path=True)


#IMPORTANT CODE BELOW

testdataframe = tc.SFrame()
test_data['image_with_predictions'] = \
    tc.object_detector.util.draw_bounding_boxes(test_data['image'],model.predict(test_data['image']))
test_data.explore()

