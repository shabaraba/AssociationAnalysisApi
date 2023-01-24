import pandas as pd
import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def handler(event, context):
    """
        request: json such as bellow:
            {
                data: [
                    ['aaa', 'bbb', 'ccc'],
                    ['aaa', 'ccc'],
                    ['bbb'],
                ],
                min_support: 0.01,
                rule: {
                    metric: "confidence",
                    min_threshold: "0.1"
                },
                condition: {
                    confidence: 0.2,
                    lift: 1.0
                }
            }
    """
    print(event)
    request = json.loads(event.get("body"))
    print(request)

    transactions = request.get('data')
    if not transactions:
        return { 'statusCode': 200, 'body': "[]" }

    # データをテーブル形式に加工
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # display(df)

    # itemsets・support算出
    freq_items = apriori(
        df,                                                # データフレーム
        min_support=float(request.get("min_support", 0.01)), # 支持度(support)の最小値
        use_colnames=True,                                 # 出力値のカラムに購入商品名を表示
        max_len=None,                                      # 生成されるitemsetsの個数
        verbose=1,                                         # low_memory=Trueの場合のイテレーション数
        low_memory=True,                                   # メモリ制限あり＆大規模なデータセット利用時に有効
    )
    sorted_freq_items = freq_items.sort_values(
        "support", ascending=False).reset_index(drop=True)
    print(sorted_freq_items)

    # アソシエーション・ルール抽出
    rule = request.get("rule")
    df_rules = association_rules(
        sorted_freq_items,                            # supportとitemsetsを持つデータフレーム
        metric=rule.get("metric", "confidence"),      # アソシエーション・ルールの評価指標
        min_threshold=float(rule.get("min_threshold", 0.1)), # metricsの閾値
    )
    print (df_rules)

    condition = request.get("condition")
    results = df_rules[
        (df_rules['confidence'] > float(condition.get("confidence", 0.2))) &  # 信頼度
        (df_rules['lift'] > float(condition.get("lift", 1.0)))  # リフト値
    ]
    print(results.loc[:,["antecedents","consequents","confidence","lift"]])
    response = json.loads(results.to_json(orient="table")).get("data")
    print(json.dumps(response))

    return {
        'statusCode': 200,
        'body': json.dumps(response),
    }


if __name__ == '__main__':
    handler(event={
        "body": json.dumps({
            "data": [
                ["aaa", "bbb", "ccc"],
                ["aaa", "ccc"],
                ["bbb"]
            ],
            "min_support": 0.01,
            "rule": {
                "metric": "confidence",
                "min_threshold": "0.1"
            },
            "condition": {
                "confidence": 0.2,
                "lift": 1.0
            }
        })
    }, context='')
