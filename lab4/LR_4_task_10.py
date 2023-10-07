import argparse
import json
import numpy as np

from LR_4_task_9 import pearson_score


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Find users who are similar to the in-put user"
    )
    parser.add_argument("--user", dest="user", required=True, help="Input user")
    return parser


# Знаходження користувачів у наборі даних, схожих на введеного користувача
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError("Cannot find " + user + " in the dataset")

    # Обчислення оцінки подібності за Пірсоном між
    # вказаним користувачем та всіма іншими
    # користувачами в наборі даних
    scores = np.array(
        [[x, pearson_score(dataset, user, x)] for x in dataset if x != user]
    )

    # Сортування оцінок за спаданням
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Вилучення оцінок перших 'num_users' користувачів
    top_users = scores_sorted[:num_users]

    return scores[top_users]


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = "ratings.json"

    with open(ratings_file, "r") as f:
        data = json.loads(f.read())

    print("\nUsers similar to " + user + ":\n")
    similar_users = find_similar_users(data, user, 3)
    print("User\t\t\tSimilarity score")
    print("-" * 41)
    for item in similar_users:
        print(item[0], "\t\t", round(float(item[1]), 2))
