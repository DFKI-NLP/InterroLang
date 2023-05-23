GRAMMAR = r"""
?start: action
action: operation done | operation join action | followup done

join: and | or
and: " and"
or: " or"
followup: " followup"
done: " [e]"

operation: adversarial | augment | cfe | data | define | featureattribution | filter | function | globaltopk | includes | keywords | label | labelfilter | lastturnfilter | lastturnop | likelihood | mistakes | modeldescription | ndatapoints | predfilter | predictions | randomprediction | rationalize | score | show | similarity | whatami

adversarial: " adversarial"

augment: " augment"

cfe: " cfe" cfefeature
cfefeature: {availablefeaturetypes} | " "

data: " data"

define: defineword allfeaturenames
defineword: " define"

featureattribution: featureattributionword (allfeaturenames | allfeaturesword | topk | attrsentence)
featureattributionword: " nlpattribute"
allfeaturesword: " all"
topk: topkword ( {topkvalues} )
topkword: " topk"
attrsentence: " sentence" 

filter: filterword featuretype
filterword: " filter"
featuretype: {availablefeaturetypes}

function: " function"

globaltopk: globaltopkword
globaltopkword: " important" (class | " all" | topk)

includes: " includes"

keywords: kwword ( {topkvalues} | allfeaturesword )
kwword: " keywords"

label: " label"

labelfilter: " labelfilter" class

lastturnfilter: " previousfilter"
lastturnop: " previousoperation"

likelihood: " likelihood"

mistakes: mistakesword mistakestypes
mistakesword: " mistake"
mistakestypes: " count" | " sample"

modeldescription: model ( modelflag )
model: " model"
modelflag: " model_name" | " model_summary" | " task" | " epochs" | " lr" | " loss" | " optimizer"

ndatapoints: " countdata"

predfilter: " predictionfilter" class

predictions: " predict"

randomprediction: " randompredict"

rationalize: " rationalize"

score: scoreword metricword (scoresetting)
scoreword: " score"
metricword: " default" | " accuracy" | " f1" | " roc" | " precision" | " recall" | " sensitivity" | " specificity" | " ppv" | " npv"
scoresetting: " micro" | " macro" | " weighted" | " "

show: " show"

similarity: similarword ( {topkvalues} )
similarword: " similar"

whatami: " self"
"""  # noqa: E501

# append the cat feature name and
# the values in another nonterminal
CAT_FEATURES = r"""
catnames: {catfeaturenames}
"""

TARGET_VAR = r"""
class: {classes}
"""

# numfeaturenames are the numerical feature names
# and numvalues are the potential numeric values
NUM_FEATURES = r"""
numnames: {numfeaturenames}
equality: gt | lt | gte | lte | eq | ne
gt: " greater than"
gte: " greater equal than"
lt: " less than"
lte: " less equal than"
eq: " equal to"
ne: " not equal to"
"""
