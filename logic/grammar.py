GRAMMAR = r"""
?start: action
action: operation done | operation join action | followup done
operation: explanation | filter | predictions | whatami | lastturnfilter | lastturnop | data | impfeatures | show | likelihood | modeldescription | function | score | ndatapoints | label | mistakes | fstats | define | labelfilter | predfilter | includes | globaltopk | cfe | similarity | rationalize | randomprediction

cfe: " cfe" cfefeature
cfefeature: {availablefeaturetypes} | " "

globaltopk: globaltopkword
globaltopkword: " important" (classname | " all" | topk)
classname: " true" | " false"

labelfilter: " labelfilter" class
predfilter: " predictionfilter" class

fstats: fstatsword (allfeaturenames | " target")
fstatsword: " statistic"

define: defineword allfeaturenames
defineword: " define"

ndatapoints: " countdata"

mistakes: mistakesword mistakestypes
mistakesword: " mistake"
mistakestypes: " count" | " sample"

label: " label"

join: and | or
and: " and"
or: " or"
filterword: " filter"

filter: filterword featuretype
featuretype: {availablefeaturetypes}

explanation: explainword explaintype
explainword: " explain"
explaintype: featureimportance
featureimportance: " features"

similarity: " similar"

rationalize: " rationalize"

predictions: " predict"
randomprediction: " randompredict"

whatami: " self"

data: " data"
modeldescription: " model"
function: " function"

score: scoreword metricword (scoresetting)
scoreword: " score"
metricword: " default" | " accuracy" | " f1" | " roc" | " precision" | " recall" | " sensitivity" | " specificity" | " ppv" | " npv"
scoresetting: " micro" | " macro" | " weighted" | " "
testword: " test"

followup: " followup"

show: " show"

likelihood: likelihoodword
likelihoodword: " likelihood"

lastturnfilter: " previousfilter"
lastturnop: " previousoperation"

impfeatures: impfeaturesword (allfeaturenames | allfeaturesword | topk | impsentence)
allfeaturesword: " all"
topk: topkword ( {topkvalues} )
topkword: " topk"
impsentence: " sentence" 


impfeaturesword: " nlpattribute"
numupdates: " increase" | " set" | " decrease"

includes: " includes"

done: " [e]"
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
