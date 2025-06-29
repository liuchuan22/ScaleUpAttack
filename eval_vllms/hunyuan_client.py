from typing import List, Dict, Any, Literal
import time
from PIL import Image

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

tencent_cloud_secretId = ""
tencent_cloud_secretKey = ""

class HunyuanChat:
    """
    Chat class for Hunyuan model from Tencent
    """
    
    def __init__(self, **kargs):
        # 实例化一个认证对象，入参需要传入腾讯云账户secretId，secretKey
        self.cred = credential.Credential(
            tencent_cloud_secretId, tencent_cloud_secretKey)

        self.cpf = ClientProfile()
        # 预先建立连接可以降低访问延迟
        self.cpf.httpProfile.pre_conn_pool_size = 3
        self.client = hunyuan_client.HunyuanClient(self.cred, "ap-guangzhou", self.cpf)

        act_req = models.ActivateServiceRequest()
        resp = self.client.ActivateService(act_req)
        self.max_retries = 10
        self.timeout = 1

    def get_single_response(self, message: List, **generation_kwargs):
        message = message[0]
        if message["role"] in ["system", "user", "assistant"]:
            msg = models.Message()
                
            msg.Contents = [models.Content(), models.Content()]
            msg.Contents[0].Type = 'text'
            msg.Contents[0].Text = message['content'][0]['text']
            msg.Contents[1].Type = 'image_url'
            imgurl = models.ImageUrl()
            imgurl.Url = f"data:image/png;base64,{message['content'][1]['source']['data']}"
            msg.Contents[1].ImageUrl = imgurl
            msg.Role = message['role']
        else:
            raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        req = models.ChatCompletionsRequest()
        req.Messages = [msg]
        req.Model = "hunyuan-vision"
        req.Stream = False

        req.Temperature = generation_kwargs.get("temperature", 1.0) if generation_kwargs.get("do_sample", True) else 0.0

        for i in range(self.max_retries):
            try:
                response = self.client.ChatCompletions(req)
                break
            except TencentCloudSDKException as err:
                print(f"Error in generation: {err}")
                response = f"Error in generation: {err}"
                time.sleep(self.timeout)
        
        assert not isinstance(response, str), response

        response_message = response.Choices[0].Message.Content
        return response_message
