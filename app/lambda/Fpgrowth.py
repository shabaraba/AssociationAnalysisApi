import numpy as np
import itertools
import ast
import ujson
from orangecontrib.associate import fpgrowth as fp
# from app.entities.Baskets import Basket


# class Fpgrowth():
#     MAX_EDGE_COUNT = 300
#     ASSOCIATION_RATE = 0.01
#     """pyfpgrowth entity
#         インスタンス化せず、createXXX系を用いて作成すること
#     """
#     def __init__(self):
#         self._basketList = []
#         self._patterns = []  # pyfpgrowthの生データ
#         self._rules = []  # 出力の際に算出
#         self._stats = []
#         self._result = []
        
#         self._field_list = []
#         self._logger: Logger

#     def __str__(self):
#         return """
#             patterns: {}件
#             rules: {}
#             stats: {}
#             result: {}件
#         """.format(len(self._patterns), self._rules, self._stats, len(self._result))

#     @staticmethod
#     def create_by_data_list(_list, _field_list, _count, _logger) -> 'Fpgrowth':
#         pyfpgrowth = Fpgrowth()
#         pyfpgrowth._field_list = _field_list
#         if _logger is not None:
#             pyfpgrowth._logger = _logger

#         # 店舗情報をlistから取り除く
#         # TODO 今後、店舗情報も分析対象に加えたい
#         _list = Fpgrowth._removeWithoutProductIdData(_list)
#         _list = Fpgrowth._removeTransactionHeadData(_list)
#         if Basket.PREFIXES_STORE not in _field_list:
#             _list = Fpgrowth._removeStoreData(_list)
#         if Basket.PREFIXES_MEMBER not in _field_list:
#             _list = Fpgrowth._removeMemberData(_list)
#         if Basket.PREFIXES_SEX not in _field_list:
#             _list = Fpgrowth._removeSexData(_list)
#         if Basket.PREFIXES_CUSTOMER_GROUP not in _field_list:
#             _list = Fpgrowth._remove_customer_group_data(_list)

#         _numberKeyDict, _columnKeyDict = pyfpgrowth._getKeyDictionaries(_list)

#         _encodedList = pyfpgrowth._encode(_list, _columnKeyDict)

#         # X, mapping = fp.OneHot.encode(_list)
#         # TODO 今後、ルールやリフト値も引数にして画面から入力できるようにしたい
#         itemsets = dict(fp.frequent_itemsets(_encodedList, 0.01))
#         pyfpgrowth._patterns = itemsets
#         # アソシエーションルールの抽出
#         rules = fp.association_rules(itemsets, 0.6)
#         pyfpgrowth._rules = rules
#         # リフト値を含んだ結果を取得
#         stats = fp.rules_stats(rules, itemsets, len(_encodedList))
#         pyfpgrowth._stats = stats
#         # リフト値（7番目の要素）でソート

#         result = []
#         if len(itemsets) > 0:
#             for s in sorted(stats, key=lambda x: x[6], reverse=True):

#                 lhs = pyfpgrowth._decode(s[0], _numberKeyDict)
#                 rhs = pyfpgrowth._decode(s[1], _numberKeyDict)

#                 support = s[2]
#                 confidence = s[3]
#                 lift = s[6]

#                 print(f"lhs = {lhs}, rhs = {rhs}, support = {support}, confidence = {confidence}, lift = {lift}")

#                 if lift < 1:
#                     break

#                 result.append(
#                     {
#                         "from": lhs,
#                         "to": rhs,
#                         "support": support,
#                         "confidence": confidence,
#                         "lift": lift
#                     }
#                 )
#         pyfpgrowth._result = result
#         return pyfpgrowth

#     @staticmethod
#     def _getKeyDictionaries(_list):
#         # バスケット内に登場する全データを重複を除いて1行リストにまとめる
#         _flatDataList = np.unique(list(itertools.chain.from_iterable(_list)))
#         # 全データにナンバリング
#         _flatDataNumberList = list(range(len(_flatDataList)))
#         _numberKeyDict = {key: value for key, value in zip(_flatDataNumberList, _flatDataList)}
#         _columnKeyDict = {key: value for value, key in _numberKeyDict.items()}
#         return _numberKeyDict, _columnKeyDict

#     @staticmethod
#     def _encode(_list, _columnKeyDict) -> list:
#         # orange3-associate用のデータ整形
#         # 全データを番号に置換
#         _encodedList = [
#             [_columnKeyDict[column] for column in _each] for _each in _list
#         ]
#         return _encodedList

#     @staticmethod
#     def _decode(_X, _numberKeyDict) -> list:
#         _result = []
#         for number in _X:
#             _idStrings = _numberKeyDict[number]
#             _convertedDataDict = Fpgrowth._converteDictFromIdString(_idStrings)
#             _result.append(_convertedDataDict)

#         return _result

#     @staticmethod
#     def _removeStoreData(_list) -> list:
#         result = []
#         for _eachBasket in _list:
#             for _each in _eachBasket:
#                 if (_each.startswith(Basket.PREFIXES_STORE)): # store__{"id": xxx}
#                     _eachBasket.remove(_each)
#             result.append(_eachBasket)
#         return result

#     @staticmethod
#     def _removeTransactionHeadData(_list) -> list:
#         result = []
#         for _eachBasket in _list:
#             for _each in _eachBasket:
#                 if (_each.startswith(Basket.PREFIXES_TRANSACTION_HEAD)): # store__{"id": xxx}
#                     _eachBasket.remove(_each)
#             result.append(_eachBasket)
#         return result

#     @staticmethod
#     def _removeMemberData(_list) -> list:
#         result = []
#         for _eachBasket in _list:
#             for _each in _eachBasket:
#                 if (_each.startswith(Basket.PREFIXES_MEMBER)): # store__{"id": xxx}
#                     _eachBasket.remove(_each)
#             result.append(_eachBasket)
#         return result

#     @staticmethod
#     def _removeSexData(_list) -> list:
#         result = []
#         for _eachBasket in _list:
#             _eachResult = []
#             for _each in _eachBasket:
#                 if not (_each.startswith(Basket.PREFIXES_SEX)): # store__{"id": xxx}
#                     _eachResult.append(_each)
#             result.append(_eachResult)
#         return result

#     @staticmethod
#     def _remove_customer_group_data(_list) -> list:
#         result = []
#         for _each_basket in _list:
#             _each_result = []
#             for _each in _each_basket:
#                 if not (_each.startswith(Basket.PREFIXES_CUSTOMER_GROUP)): # store__{"id": xxx}
#                     _each_result.append(_each)
#             result.append(_each_result)
#         return result

#     @staticmethod
#     def _removeWithoutProductIdData(_list) -> list:
#         """remove product data containing no product id
#         Arguments:
#             _list {[type]} -- [description]
#         Returns:
#             list -- [description]
#         """
#         result = []
#         for _eachBasket in _list:
#             _eachResult = []
#             for _each in _eachBasket:
#                 if not (_each.startswith(Basket.PREFIXES_PRODUCT)): # store__{"id": xxx}
#                     _eachResult.append(_each)
#                     continue
#                 _dictString = _each.replace(Basket.PREFIXES_PRODUCT, "")
#                 _json = ujson.loads(_dictString)
#                 if (_json["id"] is not None):
#                     _eachResult.append(_each)
#                     continue
#             result.append(_eachResult)
#         return result

#     @staticmethod
#     def createByPatternJson(_json):
#         _loadedList = ujson.loads(_json)
#         _patterns = {}
#         for _eachDict, val in _loadedList.items():
#             _patterns[ast.literal_eval(_eachDict)] = val

#         return Pyfpgrowth(_patterns)


#     @property
#     def result(self):
#         return self._result

#     @property
#     def patterns(self):
#         return self._patterns


#     @property
#     def stringPatterns(self):
#         return ujson.dumps(self._patterns)


#     @property
#     def rules(self):
#         return 


#     @property
#     def stringRules(self):
#         return ujson.dumps(self.rules)


#     def merge(self, _pyfpgrowthEntity):
#         """当entityに、PyfpgrowthEntityをマージします
#         同じキーの場合はvalueを加算します
#         Arguments:
#             _pyfpgrowthEntity {Pyfpgrowth} -- [description]
#         Returns:
#             [type] -- [description]
#         """
#         for key, value in _pyfpgrowthEntity.patterns.items():
#             if key not in self._patterns.keys():
#                 self._patterns[key] = 0
#             self._patterns[key] += value

#         return self

#     def convert_to_vis_js(self) -> VisJs:
#         vis = VisJs()

#         if len(self._result) <= 0:
#             self._logger.debug("debug3")
#             return vis

#         self._logger.info("---- calc max lift ----")
#         _maxLift = max([nodeGroup['lift'] for nodeGroup in self._result])

#         self._logger.info("maxLift: {}".format(_maxLift))

#         self._logger.info("nodeGroup: {}件".format(len(self._result)))
#         for nodeGroup in self._result:
#             # edgesがlimitを超えたら了
#             if (len(vis.edgeList) > self.MAX_EDGE_COUNT):
#                 break

#             # edgeから見ていく（キーの要素数が1、要素の要素数が1の場合）
#             # edgeのfrom, toで、まだnodeにない場合はnodeに格納
#             # productId=nullの場合もある。nullは未登録商品だったりテーブルチャージなので、nodeに加えない
#             for node in nodeGroup['from']:
#                 nodeFrom = node
#                 id_value = nodeFrom['id']
#                 type_prefix = nodeFrom['type_prefix']
#                 from_id = f'{type_prefix}{id_value}'
#                 if type_prefix == Basket.PREFIXES_PRODUCT:
#                     uri_query = 'productId'
#                 if type_prefix == Basket.PREFIXES_CUSTOMER_GROUP:
#                     uri_query = 'customerGroupId'
#                 if (from_id not in [node.id for node in vis.nodeList]):
#                     self._logger.info("find 'from' node id: {}".format(from_id))

#                     vis.nodeList.append(vis.Node(
#                         id=from_id,
#                         type_prefix=nodeFrom["type_prefix"],
#                         label=nodeFrom["label"],
#                         uri="https://www1.smaregi.jp/control/master/product/detail.html?productId={}".format(id_value)
#                     ))
#             for node in nodeGroup['to']:
#                 nodeTo = node
#                 id_value = nodeTo['id']
#                 type_prefix = nodeTo['type_prefix']
#                 to_id = f'{type_prefix}{id_value}'
#                 if (to_id not in [node.id for node in vis.nodeList]):
#                     self._logger.info("find 'to' node id: {}".format(to_id))
#                     vis.nodeList.append(vis.Node(
#                         id=to_id,
#                         type_prefix=nodeTo["type_prefix"],
#                         label=nodeTo["label"],
#                         uri="https://www1.smaregi.jp/control/master/product/detail.html?productId={}".format(id_value)
#                     ))
            
#             if (len(nodeGroup['from']) == 1) and (len(nodeGroup['to']) == 1):
#                 from_id_value = nodeGroup['from'][0]["id"]
#                 from_type_prefix = nodeGroup['from'][0]["type_prefix"]
#                 from_id = f'{from_type_prefix}{from_id_value}'
#                 to_id_value = nodeGroup['to'][0]["id"]
#                 to_type_prefix = nodeGroup['to'][0]["type_prefix"]
#                 to_id = f'{to_type_prefix}{to_id_value}'
#                 # 指定された分析対象の組み合わせのみvisに保存
#                 if (
#                     (
#                         from_type_prefix == self._field_list[0] and
#                         to_type_prefix == self._field_list[1]
#                     ) or
#                     (
#                         from_type_prefix == self._field_list[1] and
#                         to_type_prefix == self._field_list[0]
#                     )
#                 ):
#                     vis.edgeList.append(vis.Edge(
#                         fromNode=from_id,
#                         toNode=to_id,
#                         width=nodeGroup['lift'] / _maxLift * 5
#                     ))
#         self._logger.info("---- convertion finished ----")
#         self._logger.info(vis)
#         return vis

#     def _getDictForVis(self, data):
#         if (data.startswith(Basket.PREFIXES_PRODUCT)):
#             dataJson = data.split(Basket.PREFIXES_PRODUCT)[1]
#             dataDict = ujson.loads(dataJson)
#             return {
#                 "id":dataDict["id"],
#                 "label": self._productsApi.getProductById(dataDict["id"])["productName"]
#             }
#         # elif (data.startswith('customerSex__')): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#         #     customerSexJson = data.split('customerSex__')[1]
#         #     customerSexDict = ujson.loads(customerSexJson)
#         #     nodeId = customerSexDict['sex']
#         #     nodeLabel = customerSexDict['sex']

#         #     return {
#         #         "id"   : nodeId,
#         #         "label": nodeLabel,
#         #     }
#         # elif (data.startswith('store__')): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#         #     storeJson = data.split('store__')[1]
#         #     storeDict = ujson.loads(storeJson)
#         #     nodeId = storeDict["id"]
#         #     nodeLabel = storeDict["id"]

#         #     return {
#         #         "id"   : nodeId,
#         #         "label": nodeLabel,
#         #     }
#         # elif (data.startswith('member__')): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#         #     memberJson = data.split('member__')[1]
#         #     memberDict = ujson.loads(memberJson)
#         #     nodeId = memberDict["id"]
#         #     nodeLabel = memberDict["id"]

#         #     return {
#         #         "id"   : nodeId,
#         #         "label": nodeLabel,
#         #     }
#         else:
#             return None


#     @staticmethod
#     def _converteDictFromIdString(data) -> dict:
#         # _productsRepository = ProductsRepository()
#         if (data.startswith(Basket.PREFIXES_PRODUCT)):
#             dataJson = data.split(Basket.PREFIXES_PRODUCT)[1]
#             dataDict = ujson.loads(dataJson)
#             return {
#                 "id": dataDict["id"],
#                 # "label": _productsRepository.getProductById(dataDict["id"]).name
#                 "label": dataDict['id'],
#                 "type_prefix": Basket.PREFIXES_PRODUCT
#             }
#         elif (data.startswith(Basket.PREFIXES_SEX)): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#             customerSexJson = data.split(Basket.PREFIXES_SEX)[1]
#             customerSexDict = ujson.loads(customerSexJson)
#             nodeId = customerSexDict['sex']
#             nodeLabel = customerSexDict['sex']

#             return {
#                 "id": nodeId,
#                 "label": nodeLabel,
#                 "type_prefix": Basket.PREFIXES_SEX
#             }
#         elif (data.startswith(Basket.PREFIXES_STORE)): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#             storeJson = data.split(Basket.PREFIXES_STORE)[1]
#             storeDict = ujson.loads(storeJson)
#             nodeId = storeDict["id"]
#             nodeLabel = storeDict["id"]

#             return {
#                 "id": nodeId,
#                 "label": nodeLabel,
#                 "type_prefix": Basket.PREFIXES_STORE
#             }
#         elif (data.startswith(Basket.PREFIXES_MEMBER)): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#             memberJson = data.split(Basket.PREFIXES_MEMBER)[1]
#             memberDict = ujson.loads(memberJson)
#             nodeId = memberDict["id"]
#             nodeLabel = memberDict["id"]

#             return {
#                 "id": nodeId,
#                 "label": nodeLabel,
#                 "type_prefix": Basket.PREFIXES_MEMBER
#             }
#         elif (data.startswith(Basket.PREFIXES_CUSTOMER_GROUP)): # product__{"id": xxx, "name": xxx, "categoryId": xxx}
#             customer_group_json = data.split(Basket.PREFIXES_CUSTOMER_GROUP)[1]
#             customer_group_dict = ujson.loads(customer_group_json)
#             node_id = customer_group_dict["id"]
#             node_label = customer_group_dict["id"]

#             return {
#                 "id": node_id,
#                 "label": node_label,
#                 "type_prefix": Basket.PREFIXES_CUSTOMER_GROUP
#             }
#         else:
#             return None