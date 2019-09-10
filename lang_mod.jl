using CorpusLoaders
using Embeddings
using Flux
using Flux: onehot, onehotbatch
using DifferentialEquations
using DiffEqFlux
using MultiResolutionIterators
using WordTokenizers
using WordTokenizers: TokenBuffer, number

WORD_EMBED_DIMS = 50

train_set = load(CoNLL(), "train") # training set
dataset = flatten_levels(train_set, lvls(CoNLL, :document)) |> full_consolidate
X_train = [CorpusLoaders.word.(sent) for sent in dataset]
X = sort(X_train, alg=MergeSort, by=length, rev=true)
X = X[11:4010] # Just remove too long and too short sents

## GloVe Word Embeddings

embtable = load_embeddings(GloVe, 1)
get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))
get_word_from_index = Dict(value => key for (key, value) in get_word_index)
W_word_Embed = (embtable.embeddings)

UNK_Word = "<UNK>"
UNK_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert UNK_Word ∉ collect(keys(get_word_index))

get_word_index[UNK_Word] = UNK_Word_Idx
get_word_from_index[UNK_Word_Idx] = UNK_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))

# One Vec for END chars
PAD_Word = "<END>"
PAD_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert PAD_Word ∉ collect(keys(get_word_index))
get_word_index[PAD_Word] = PAD_Word_Idx
get_word_from_index[PAD_Word_Idx] = PAD_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))

W_word_Embed = Float32.(W_word_Embed)
@assert size(W_word_Embed, 2) == embedding_vocab_length

NUM_Word = "0"
NUM_Word_idx = get_word_index[NUM_Word]
# @assert NUM_Word ∈ collect(keys(get_word_index))

function idxword(word, oh=true)
    if (oh)
        number(TokenBuffer(word), check_sign=true) && return onehot(NUM_Word_idx, 1:embedding_vocab_length)
        return onehot(get(get_word_index, word, get(get_word_index, lowercase(word), UNK_Word_Idx)), 1:embedding_vocab_length)
    end

    number(TokenBuffer(word), check_sign=true) && return NUM_Word_idx
    get(get_word_index, word, get(get_word_index, lowercase(word), UNK_Word_Idx))
end

ohb_word(words) = push!([idxword(word) for word in words], idxword(PAD_Word))#, 1:embedding_vocab_length)

X_input = [ohb_word(words) for words in X]

#################### MODEL ####################
