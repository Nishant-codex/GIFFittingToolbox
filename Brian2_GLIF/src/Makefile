GLIF_NUM := 1 2 3 4 5

GLIF_MODELS := $(addprefix GLIF_,$(addsuffix .py,$(GLIF_NUM)))
GLIF_ALLENSDK := $(addprefix GLIF_,$(addsuffix _AllenSDK.py,$(GLIF_NUM)))
GLIF_EVALUATOR := $(addprefix GLIF_,$(addsuffix _Evaluator.py,$(GLIF_NUM)))

ALL_TARGETS := $(GLIF_MODELS) $(GLIF_ALLENSDK) $(GLIF_EVALUATOR)

.PHONY: all
all: $(ALL_TARGETS)

# BluePyOpt evaluators
$(GLIF_EVALUATOR): GLIF_%_Evaluator.py: GLIF_%.hm GLIF_Evaluator_template.pym
	cmod -g -o $@ $^

# Read AllenSDK files and compare results
$(GLIF_ALLENSDK): GLIF_%_AllenSDK.py: GLIF_%.hm GLIF_%_AllenSDK.hm GLIF_AllenSDK_template.pym
	cmod -g -o $@ $^

# Brian2 models
$(GLIF_MODELS): GLIF_%.py: GLIF_%.hm GLIF_template.pym
	cmod -g -o $@ $^

PREFIX := ../

.PHONY: install
install: all
	mv -f $(ALL_TARGETS) $(PREFIX)

.PHONY: clean
clean:
	$(RM) $(ALL_TARGETS)
