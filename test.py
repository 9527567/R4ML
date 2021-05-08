import pandas as pd
import numpy as np
test_dict = {'id': [1, 2, 3, 4, 5, 5, 5], 'name': ['Alice', 'Bob', 'Cindy', 'Eric', 'Helen',
                                                   'Grace ', 'haha'], 'math': [90, 89, 99, 78, 97, 93, 50], 'english': [89, 94, 80, 94, 94, 90, 20]}
my_df = pd.DataFrame(test_dict)

for i in my_df.columns:
    print(i, np.where(my_df.duplicated(subset=i) == True)[0])





haha()
