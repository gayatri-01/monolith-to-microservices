import traceback

from common import common
from extractor import Extractor
import os
import csv

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'

CSV_FILE = "D:\Gayatri\BITS WILP\Dissertation\POC2\code_vectors_blog.csv"


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    code_vectors = []  # List to store function vectors
    function_names = []  # List to store function names

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()


    def getFunctionName(self, file_path):
        # Extract the filename without extension
        file_name = os.path.basename(file_path).replace('.java', '')

        # Extract function name (assuming format: ClassName_FunctionName)
        if "_" in file_name:
            function_name = file_name.split("_", 1)[1]  # Get the part after the first underscore
        else:
            function_name = file_name  # If no underscore, assume entire name is function

        print("Extracted Function Name:", function_name)

        return function_name

    def predict(self):
        # input_filenames = [
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_createUser.java", 
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_getAllUsers.java", 
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_createProduct.java", 
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_getAllProducts.java", 
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_placeOrder.java",
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_getAllOrders.java", 
        #     "D:\Gayatri\BITS WILP\Dissertation\POC2\extracted_methods\ECommerceService_processPayment.java"
        # ]
        input_filenames = ["D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_getSiteName.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_setSiteName.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_getPageSize.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_setPageSize.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_getSiteSlogan.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\AppSetting_setSiteSlogan.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\CacheSettingService_get.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\CacheSettingService_get.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\CacheSettingService_put.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getPost.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getPublishedPostByPermalink.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_createPost.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_updatePost.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_deletePost.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getArchivePosts.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getPostTags.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_extractPostMeta.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getAllPublishedPostsByPage.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_createAboutPage.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_parseTagNames.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_getTagNames.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_findPostsByTag.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_countPostsByTags.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\PostService_incrementViews.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SettingService_get.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SettingService_get.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SettingService_put.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\TagService_findOrCreateByName.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\TagService_getTag.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\TagService_deleteTag.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\TagService_getAllTags.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_initialize.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_createUser.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_getSuperUser.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_loadUserByUsername.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_currentUser.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_changePassword.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_signin.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_authenticate.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_createSpringUser.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\UserService_createAuthority.java","D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SecurityConfig_rememberMeServices.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SecurityConfig_passwordEncoder.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SecurityConfig_configure.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\SecurityConfig_configure.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_addInterceptors.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_addCorsMappings.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_registerJadeViewHelpers.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_preHandle.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_postHandle.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_viewObjectAddingInterceptor.java", "D:\\Gayatri\\BITS WILP\\Dissertation\\SpringBlog2\\extracted_methods\\WebConfig_getApplicationEnv.java"]

        for input_filename in input_filenames:
            print('Starting interactive prediction for ...'+input_filename)
     
            # print(
            #     'Modify the file: "%s" and press any key when ready, or "q" / "quit" / "exit" to exit' % input_filename)
            # user_input = input()
            # if user_input.lower() in self.exit_keywords:
            #     print('Exiting...')
            #     return
            try:
                predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(input_filename)
            except ValueError as e:
                print(e)
                continue
            raw_prediction_results = self.model.predict(predict_lines)
            method_prediction_results = common.parse_prediction_results(
                raw_prediction_results, hash_to_string_dict,
                self.model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)
            for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
                # print('Original name:\t' + method_prediction.original_name)
                # for name_prob_pair in method_prediction.predictions:
                #     print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))
                # print('Attention:')
                # for attention_obj in method_prediction.attention_paths:
                #     print('%f\tcontext: %s,%s,%s' % (
                #     attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))
                if self.config.EXPORT_CODE_VECTORS:
                    print('Code vector:')
                    print(','.join(map(str, raw_prediction.code_vector)))
                    vector=list(raw_prediction.code_vector)
                    function_name = self.getFunctionName(input_filename)
                        # Append function name and vector to CSV
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([function_name] + vector)  # Store function name + vector

                    print(f"Exported vector for function: {function_name}")
