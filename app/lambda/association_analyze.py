import packages.pandas as pd
import json
from packages.mlxtend.preprocessing import TransactionEncoder
from packages.mlxtend.frequent_patterns import apriori
from packages.mlxtend.frequent_patterns import association_rules
from packages.IPython.display import display


def handler(event, context):
    print('event:')
    print(event)
    print('context:')
    print(context)
    request = {
        'data': [
            ['aaa', 'bbb', 'ccc'],
            ['aaa', 'ccc'],
            ['bbb'],
        ]

    }
    transactions = request.get('data')

    # データをテーブル形式に加工
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    # display(df)

    # itemsets・support算出
    freq_items = apriori(
        df,                     # データフレーム
        min_support=0.01,    # 支持度(support)の最小値
        use_colnames=True,    # 出力値のカラムに購入商品名を表示
        max_len=None,    # 生成されるitemsetsの個数
        verbose=1,            # low_memory=Trueの場合のイテレーション数
        low_memory=True,     # メモリ制限あり＆大規模なデータセット利用時に有効
    )
    sorted_freq_items = freq_items.sort_values(
        "support", ascending=False).reset_index(drop=True)
    print(sorted_freq_items)

    # アソシエーション・ルール抽出
    df_rules = association_rules(
        sorted_freq_items,             # supportとitemsetsを持つデータフレーム
        metric="confidence",  # アソシエーション・ルールの評価指標
        min_threshold=0.1,    # metricsの閾値
    )
    print (df_rules)

    results = df_rules[
        (df_rules['confidence'] > 0.2) &  # 信頼度
        (df_rules['lift'] > 1.0)  # リフト値
    ]
    print(results.loc[:,["antecedents","consequents","confidence","lift"]])
    print(results.T.to_json())

    response = {'resp': 'hello!'}

    # TODO:
    # df = pd.read_json(response)

    return {
        'statusCode': 200,
        'body': json.dumps(response),
    }


if __name__ == '__main__':
    handler(event='', context='')
